import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull
from math import pi,sqrt
with gzip.open('data/ab.txt.gz','rt') as f:
    S,D,H,n=map(int,f.readline().split())
    points=[];section=[]
    for i in range(n):
        z,k=map(int,f.readline().split())
        xy=np.array([list(map(int,f.readline().split())) for _ in range(k)])*[1/S,sqrt(3)/S]
        z*=sqrt(6)/S
        axis=np.array([sqrt(3)/2,.5]);perp=np.array([-.5,sqrt(3)/2])
        r=xy@axis;t=xy@perp
        rr=np.roll(r,-1);tt=np.roll(t,-1);jj=(tt*t<0)
        if jj.sum()==2:
            section.append([z,*sorted((r[jj]-t[jj]*(rr[jj]-r[jj])/(tt[jj]-t[jj])))])
        if i%6==0 or i==n-1:
            p=xy[::max(1,k//100)]
            points.extend(np.c_[p,np.full(len(p),z)])
p=np.array(points);h=ConvexHull(p)
fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
view=np.array([.76,.45,.46]);view/=np.linalg.norm(view)
ex=np.cross([0,0,1],view);ex/=np.linalg.norm(ex);ey=np.cross(view,ex)
norm=h.equations[:,:3];front=norm@view>0
xy=np.c_[p@ex,p@ey];faces=xy[h.simplices[front]]
light=np.array([-.2,-.45,.87])
rgb=np.array([.22,.57,.62])*(.65+.41*(norm[front]@light))[:,None]
axes[0].add_collection(PolyCollection(faces,facecolors=rgb,edgecolors=rgb,linewidths=.45))
axes[0].autoscale();axes[0].set_aspect('equal');axes[0].axis('off');axes[0].set_title('One component')
z,l,r=np.array(section).T
ax=axes[1];ax.fill_betweenx(z,l,r,color='#79aeb0',alpha=.75,label='Constructed body')
phi=np.linspace(0,2*pi,700);ax.plot(.5*np.cos(phi),.5*np.sin(phi),'--',color='#584a42',lw=1,label='Circle of radius 1/2')
ax.set_aspect('equal');ax.set_xlabel('Horizontal coordinate at azimuth 30°');ax.set_ylabel('Height');ax.legend(loc='lower center',bbox_to_anchor=(.5,-.38),frameon=False);ax.set_title('Vertical section')
fig.savefig('figures/body.png',dpi=180)
