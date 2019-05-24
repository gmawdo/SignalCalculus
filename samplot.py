import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File

# Tile9050nbrsRadius00_75thresh0_001vSpeed02_00dec00_10.las
inFile = File("attrTile9NFLClip100_00N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")

# create matrix of all the eigenvectors
point_focus = inFile.lang<0.3
lang = inFile.lang[point_focus]
iso = inFile.iso[point_focus]
u1,i1,c1 = np.unique(np.stack((0.05*np.floor(iso/0.05),0.05*np.floor(lang/0.05))), axis=1,return_inverse=True,return_counts=True)
c1=c1/sum(c1)

condition = (u1[0,i1]>0.85)*(u1[1,i1]<0.3)

fig = plt.figure()
print(min(c1),max(c1))
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=16, azim=43)

ax.set_xlabel('Lin. ang.')
ax.set_ylabel('Iso.')
ax.set_zlabel('Num. points')

ax.set_xlim3d(0.0, 1.0)
ax.set_ylim3d(0.0, 1.0)
ax.set_zlim3d(0, max(c1))

verts = []
u2,i2,c2 = np.unique(u1[0,:],return_inverse=True,return_counts=True)
zs = u2
facecolors = []
for z in zs:
	u = u1[:,] #this predicate extracts entries only for this entropy value
	xs = np.concatenate([[0],u1[1,u1[0,:]==z],[0]])
	ys = np.concatenate([[0],c1[u1[0,:]==z],[0]])
	#print xs,ys
	verts.append(list(zip(xs,ys)))
	facecolors.append((z, z * z, 0.0, 0.6))

poly = PolyCollection(verts, facecolors=facecolors)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

outFile = File("SamPlot.las", mode = "w", header = inFile.header)
outFile.points = inFile.points
Classn = 0*inFile.classification
classn = 0*inFile.classification[point_focus]
classn[condition]=2
Classn[point_focus] = classn
outFile.classification = Classn
print(sum(point_focus),sum(condition))
outFile.close()


plt.show()
