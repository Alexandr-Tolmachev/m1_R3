"""Exact verification of two periodic unit-distance-avoiding constructions.

Only Python's standard library is required. All geometric calculations use
arbitrary-precision integers. Run python3 verify.py to check both data files.
"""
import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class CertificateError(ValueError):
    """A certificate is malformed or violates a required inequality."""


def require(condition, message):
    if not condition:
        raise CertificateError(message)


def dot(first, second):
    return first[0] * second[0] + 3 * first[1] * second[1] + 6 * first[2] * second[2]


def squared_distance(first, second):
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    dz = first[2] - second[2]
    return dx * dx + 3 * dy * dy + 6 * dz * dz


def turn(first, second, third):
    return ((second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0]))


def convex_hull(points):
    """Return the counterclockwise boundary without redundant collinear points."""
    ordered = sorted(set(points))
    require(len(ordered) >= 3, "A section needs three distinct points.")
    chains = []
    for sequence in (ordered, reversed(ordered)):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return chains[0] + chains[1]


def twice_area(points):
    hull = convex_hull(points)
    area = sum(turn(hull[0], hull[index], hull[index + 1])
               for index in range(1, len(hull) - 1))
    require(area > 0, "A section has zero area.")
    return area


def read_integers(stream, count):
    fields = stream.readline().split()
    require(len(fields) == count, f"Expected a row of {count} integers.")
    try:
        return tuple(int(field) for field in fields)
    except ValueError as error:
        raise CertificateError("Non-integer coordinate.") from error


def load_points(path):
    with gzip.open(path, "rt", encoding="ascii") as stream:
        scale, spacing, height, count = read_integers(stream, 4)
        require(scale > 0 and spacing > 0 and height > 0, "Nonpositive parameter.")
        require(spacing % 6 == 0, "The horizontal spacing must be divisible by six.")
        require(count >= 2, "At least two sections are required.")
        sections = []
        for _ in range(count):
            level, size = read_integers(stream, 2)
            require(size >= 3, "A section needs at least three points.")
            if sections:
                require(level > sections[-1][0], "Section heights must increase.")
            polygon = [read_integers(stream, 2) for _ in range(size)]
            sections.append((level, polygon))
        require(not stream.read().strip(), "Unexpected data after the last section.")
    return scale, spacing, height, sections


def neighbour_offsets(spacing, height):
    horizontal = [(spacing, 0, 0),
                  (spacing // 2, spacing // 2, 0),
                  (-spacing // 2, spacing // 2, 0)]
    horizontal += [tuple(-x for x in point) for point in horizontal]
    interlayer = [(spacing // 2, spacing // 6),
                  (-spacing // 2, spacing // 6),
                  (0, -spacing // 3)]
    return horizontal + [(x, y, z)
                         for z in (-height, height) for x, y in interlayer]


def check_separation(points, scale, spacing, height):
    """Check all twelve separating planes, the radius and the distant layers."""
    limit = scale * scale
    far_distances = (3 * spacing * spacing,
                     4 * spacing * spacing // 3 + 6 * height * height,
                     24 * height * height)
    require(all(25 * value > 121 * limit for value in far_distances),
            "Distant centres fail the separation condition.")
    offsets = [(v, dot(v, v)) for v in neighbour_offsets(spacing, height)]
    for point in points:
        require(25 * dot(point, point) < 9 * limit, "The radius bound fails.")
        for vector, norm in offsets:
            gap = norm - 2 * dot(vector, point)
            require(gap >= 0 and gap * gap >= norm * limit,
                    "A separating plane condition fails.")


@dataclass
class Box:
    lower: tuple
    upper: tuple
    size: int
    points: tuple = ()
    left: object = None
    right: object = None


def build_boxes(points, leaf_size=18):
    lower = tuple(min(point[axis] for point in points) for axis in range(3))
    upper = tuple(max(point[axis] for point in points) for axis in range(3))
    box = Box(lower, upper, len(points))
    if len(points) <= leaf_size:
        box.points = tuple(points)
        return box
    axis = max(range(3), key=lambda index: upper[index] - lower[index])
    ordered = sorted(points, key=lambda point: point[axis])
    middle = len(ordered) // 2
    box.left = build_boxes(ordered[:middle], leaf_size)
    box.right = build_boxes(ordered[middle:], leaf_size)
    return box


def check_box_pair(first, second, limit):
    """Cover every pair, pruning only when an integer upper bound proves it safe."""
    differences = tuple(max(first.upper[axis] - second.lower[axis],
                            second.upper[axis] - first.lower[axis])
                        for axis in range(3))
    if dot(differences, differences) <= limit:
        return
    if first.left is None and second.left is None:
        for point in first.points:
            for other in second.points:
                require(squared_distance(point, other) <= limit,
                        "The diameter bound fails.")
    elif first is second:
        check_box_pair(first.left, first.left, limit)
        check_box_pair(first.left, first.right, limit)
        check_box_pair(first.right, first.right, limit)
    elif second.left is None or (first.left is not None and first.size >= second.size):
        check_box_pair(first.left, second, limit)
        check_box_pair(first.right, second, limit)
    else:
        check_box_pair(first, second.left, limit)
        check_box_pair(first, second.right, limit)


def check_diameter(points, scale):
    require(bool(points), "Empty point list.")
    boxes = build_boxes(points)
    check_box_pair(boxes, boxes, scale * scale)


def volume_certificate(sections):
    areas = [twice_area(polygon) for _, polygon in sections]
    numerator = 0
    for index in range(len(sections) - 1):
        thickness = sections[index + 1][0] - sections[index][0]
        require(thickness > 0, "Section heights must increase.")
        first, second = areas[index:index + 2]
        harmonic = 2 * first * second // (first + second)
        numerator += thickness * (first + second + harmonic)
    return numerator


def certify(path):
    """Verify geometry and compute a certificate without using reference values."""
    scale, spacing, height, sections = load_points(path)
    points = [(x, y, level) for level, polygon in sections for x, y in polygon]
    check_separation(points, scale, spacing, height)
    check_diameter(points, scale)
    numerator = volume_certificate(sections)
    denominator = 3 * spacing * spacing * height
    return {
        "S": scale, "D": spacing, "H": height,
        "sections": len(sections), "points": len(points),
        "N": numerator, "M": denominator,
        "lower_numerator": numerator * 10**9 // denominator,
        "lower_denominator": 10**9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("construction", nargs="?", choices=("hcp", "ab", "all"), default="all")
    args = parser.parse_args()
    reference = json.loads((ROOT / "certificates.json").read_text())
    names = ("hcp", "ab") if args.construction == "all" else (args.construction,)
    for name in names:
        path = ROOT / "data" / f"{name}.txt.gz"
        expected = reference[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected["sha256"], f"{name}: SHA-256 mismatch.")
        print(f"Checking {name}...", flush=True)
        computed = certify(path)
        require(computed == expected["certificate"], f"{name}: computed values differ.")
        print(json.dumps(computed, indent=2))
        digits = computed["lower_numerator"]
        print(f"m_1(R^3) >= {digits // 10**9}.{digits % 10**9:09d}", flush=True)


if __name__ == "__main__":
    main()
