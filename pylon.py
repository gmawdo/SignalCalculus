import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File

inFile = File("attrTile9NFLClip100_00N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")

# inFile.x
# create matrix of all the eigenvectors
#print(max(np.round(inFile.lang,0)),min(np.round(inFile.lang)))
result1 = np.stack((np.round(inFile.iso,3),np.round(inFile.lang,2)), axis=-1)
print("formed main result set size:",len(result1))
print(result1.shape)

# find how many unique isotropies in combo with linear angles
u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
c1 = c1.reshape(u1.shape[0],1)
u1 = np.concatenate((u1,c1),axis=1)
#condition = (u1[:,2]>20) & (u1[:,2]<750)
#condition = (u1[:,1]<0.2)*(u1[:,2]>50)
#u1 = u1[condition]
u1.sort
# NOW eigen, entropy, cnt
print("form unqiue result set of distinct isotropies and langs combos with counts",u1.shape)
print(u1.shape) #total number of tuples

# for info purposes get unqiue number of isotropies and unique number of linear angles
u2,i2,c2 = np.unique(u1[:,[0]], axis=0,return_inverse=True,return_counts=True)
u3,i3,c3 = np.unique(u1[:,[1]], axis=0,return_inverse=True,return_counts=True)
print("number of distinct isotropies:",u2.shape)
u2.sort
print("number of distinct linear angles:",u3.shape)

#isotropies,linear angle, cnt
x, y, z = u1[:,0], u1[:,1], u1[:,2]

#classify a set of points into a new file
#outFile = File("Gary-pylon.las", mode = "w", header = inFile.header)
#classification = inFile.classification
#outFile.points = inFile.points
#
#
#writecondition = (inFile.ent==0.000000001)
#classification[np.logical_not(writecondition)]=0
#
#classi = 10
#for s in u3[:,0]:
#	writecondition = (np.round(inFile.ent,3)==s)
#	classification[writecondition]=classi
#	classi = classi + 1
#
#outFile.classification = classification
#outFile.close()

inFile.close()
#PLOT
#PLOT
fig = plt.figure()

#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=16, azim=43)

ax.set_xlabel('isotropy')
ax.set_ylabel('lang')
ax.set_zlabel('num points')

ax.set_xlim3d(0.0, 1.0) #iso
ax.set_ylim3d(0.0, 1.0) #lang
ax.set_zlim3d(0, 15000) #numpoints

#ax.set_xlim3d(0, 200)
#ax.set_ylim3d(0.0, 0.03)
#ax.set_zlim3d(0, 500)
verts = []
zs = u3[:,0]
facecolors =[]
for z in u3[:,0]: #for each linear angle
    ux = u1[(u1[:,1]==z)] #this predicate extracts entries only for this lang value
    xs = np.concatenate([[0],ux[:,0],[0]]) #iso
    ys = np.concatenate([[0],ux[:,2],[0]]) #num points
    #print(xs)
    #print("****")
    #print(ys)
    verts.append(list(zip(xs,ys)))
    facecolors.append((z, z * z, 0.0, 0.6))

poly = PolyCollection(verts, facecolors=facecolors)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

#X, Y = np.meshgrid(x, y, sparse=True)
#Z = z.reshape(X.shape)
#ax.plot_surface(X, Y, Z)

plt.show()