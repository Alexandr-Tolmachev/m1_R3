"""Projection of two layers and the separating strip for an interlayer neighbour."""
import gzip
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

blue, red = '#116575', '#b25c37'
with gzip.open('data/ab.txt.gz','rt') as stream:
    scale, d, h, count = map(int,stream.readline().split())
    d, h = d/scale, sqrt(6)*h/scale
    profile = []
    for _ in range(count):
        z, n = map(int,stream.readline().split())
        xy = np.array([list(map(int,stream.readline().split())) for _ in range(n)]) * [1/scale,sqrt(3)/scale]
        u = xy @ [sqrt(3)/2, .5]
        t = xy @ [-.5,sqrt(3)/2]
        v, s = np.roll(u,-1), np.roll(t,-1)
        cross = t*s < 0
        if cross.sum()==2:
            ends = u[cross] + (v[cross]-u[cross])*t[cross]/(t[cross]-s[cross])
            profile.append((min(ends),max(ends),sqrt(6)*z/scale))
fig,(plan,cut) = plt.subplots(1,2,figsize=(8.2,3.2),layout='constrained')
phi = np.arange(7)*pi/3
ring = d*np.column_stack([np.cos(phi),np.sin(phi)])
plan.plot(*ring.T,color='#c3d3d6',lw=1)
for node in ring[:-1]:
    plan.plot([0,node[0]],[0,node[1]],color='#c3d3d6',lw=1)
plan.scatter(*ring[:-1].T,color=blue,s=52)
plan.scatter(0,0,color=blue,s=52,label='Layer A')
ang = pi/6+np.arange(3)*2*pi/3
plan.scatter(d*np.cos(ang)/sqrt(3),d*np.sin(ang)/sqrt(3),color=red,marker='^',s=64,label='Layer B')
plan.legend(loc='lower center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
plan.set_title('Projection of centres')
low,high,z = np.array(profile).T
outline = np.column_stack([np.r_[low,high[::-1]],np.r_[z,z[::-1]]])
offset = np.array([d/sqrt(3),h])
normal = offset/np.linalg.norm(offset)
along = (np.linalg.norm(offset)-1)*.5
tangent = np.array([-normal[1],normal[0]])
for foot in (along*normal, offset-along*normal):
    line = foot+np.outer([-.55,.55],tangent)
    cut.plot(*line.T,'--',color='#58656c',lw=1)
for vertices, color in ((outline,blue),(offset-outline,red)):
    cut.fill(*vertices.T,color=color,alpha=.85)
cut.annotate('',xy=offset-along*normal,xytext=along*normal,arrowprops=dict(arrowstyle='<->',color='#273337'))
cut.text(*(offset/2+.14*tangent),'1',ha='center',va='center')
cut.set_title('Section through the centres')
for ax in (plan,cut):
    ax.set_aspect('equal')
    ax.axis('off')
fig.savefig('figures/placement.png',dpi=200,bbox_inches='tight')
