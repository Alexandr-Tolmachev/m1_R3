"""Small independent checks for the exact verifier. Run python3 -m unittest -v."""
import gzip
from pathlib import Path
import random
import tempfile
import unittest

from verify import (
    CertificateError, check_diameter, check_separation, load_points,
    squared_distance, twice_area, volume_certificate,
)


class VerificationTests(unittest.TestCase):
    def test_box_algorithm_against_exhaustive_distances(self):
        random_source = random.Random(117)
        for _ in range(20):
            points = [tuple(random_source.randrange(-30, 31) for _ in range(3))
                      for _ in range(45)]
            maximum = max(squared_distance(p, q) for p in points for q in points)
            for scale in range(1, 200, 7):
                if maximum <= scale * scale:
                    check_diameter(points, scale)
                else:
                    with self.assertRaises(CertificateError):
                        check_diameter(points, scale)

    def test_touching_diameter_and_one_integer_violation(self):
        check_diameter([(0, 0, 0), (10, 0, 0)], 10)
        with self.assertRaises(CertificateError):
            check_diameter([(0, 0, 0), (11, 0, 0)], 10)

    def test_polygon_area_with_unordered_redundant_points(self):
        points = [(4, 3), (0, 0), (2, 0), (0, 3), (4, 0), (1, 1), (0, 0)]
        self.assertEqual(twice_area(points), 24)
        with self.assertRaises(CertificateError):
            twice_area([(0, 0), (1, 1), (2, 2)])

    def test_volume_of_prism(self):
        square = [(0, 0), (3, 0), (3, 3), (0, 3)]
        self.assertEqual(volume_certificate([(0, square), (5, square)]), 6 * 45)

    def test_plane_radius_and_distant_centres(self):
        check_separation([(0, 0, 0)], 1_000_000, 1_912_002, 637_334)
        for points, spacing, height in [
            ([(500_000, 0, 0)], 1_912_002, 637_334),
            ([(700_000, 0, 0)], 1_912_002, 637_334),
            ([(0, 0, 0)], 1_000_002, 333_334),
        ]:
            with self.assertRaises(CertificateError):
                check_separation(points, 1_000_000, spacing, height)

    def test_truncated_reordered_and_trailing_input(self):
        header = "1000000 1912002 637334 2\n"
        first = "0 3\n0 0\n1 0\n0 1\n"
        second = "1 3\n0 0\n1 0\n0 1\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gz"
            for contents in [header + first, header + second + first,
                             header + first + second + "1\n"]:
                with gzip.open(path, "wt") as stream:
                    stream.write(contents)
                with self.assertRaises(CertificateError):
                    load_points(path)


if __name__ == "__main__":
    unittest.main()
