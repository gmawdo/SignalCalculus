import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File

def spot(x,y,z):
    A = np.stack((x,y,z), axis=-1)
    condition = ((A[:,0] >= 0) & (A[:,1] < 0) & (A[:,2] < 0)) | ((A[:,0]< 0) & (A[:,1] < 0) & (A[:,2] < 0)) | ((A[:,0] >= 0) & (A[:,1] >= 0) & (A[:,2] < 0)) | ((A[:,0] < 0) & (A[:,1] >= 0) & (A[:,2] < 0))
    A[condition] = A[condition]*-1
    return A

def cart2sph(x,y,z):
    A = spot(x,y,z)
    
    XsqPlusYsq = A[:,0]**2 + A[:,1]**2
    r = np.sqrt(XsqPlusYsq + A[:,2]**2)               # r
    elev = np.round(np.arctan2(A[:,2],np.sqrt(XsqPlusYsq)),1)     # theta
    az = np.round(np.arctan2(A[:,1],A[:,0]),1) # phi
    ans = np.absolute(elev*1000+az)
    return ans

#inFile = File("Tile9050nbrsRadius00_75thresh0_001vSpeed02_00dec00_10NFLClip100_00.las", mode = "r")
inFile = File("OLD-TEST-FILES/attrTile9NFLClip100_00N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")


print("file read")
# inFile.x
# create matrix of all the eigenvectors
result1 = np.stack((cart2sph(inFile.lambda_x,inFile.lambda_y,inFile.lambda_z),np.round(inFile.ent,3)), axis=-1)
print("formed main result set size:",len(result1))

# find how many unique eigenvectors in combo with entropy
u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
c1 = c1.reshape(u1.shape[0],1)
u1 = np.concatenate((u1,c1),axis=1)

#condition = (u1[:,2]>50) & (u1[:,2]<750)
#condition = (u1[:,1]<0.15)*(u1[:,2]>75)
condition = (u1[:,1]<0.02)
u1 = u1[condition]
u1.sort
# NOW eigen, entropy, cnt
print("form unqiue result set of distinct eignevector and entropy combos with counts",u1.shape)
print(u1.shape) #total number of tuples

# for info purposes get unqiue number of EVs and unique number of entropies
u2,i2,c2 = np.unique(u1[:,[0]], axis=0,return_inverse=True,return_counts=True) #ev
u3,i3,c3 = np.unique(u1[:,[1]], axis=0,return_inverse=True,return_counts=True) #ent

print("number of distinct EVs:",u2.shape)
u2.sort
print("number of distinct entropies:",u3.shape)

#eigenID,entropy, cnt
x, y, z = u1[:,0], u1[:,1], u1[:,2]

#classify a set of points into a new file
outFile = File("Gary.las", mode = "w", header = inFile.header)
classification = inFile.classification
outFile.points = inFile.points


writecondition = (inFile.ent==0.000000001)
classification[np.logical_not(writecondition)]=0

classi = 10
for s in u3[:,0]:
    writecondition = (np.round(inFile.ent,3)==s)
    if classi != 10:
        classification[writecondition]=classi
    classi = classi + 1
    #print("complete class %i"%classi)

outFile.classification = classification
outFile.close()

inFile.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=16, azim=43)

ax.set_xlabel('eigenvector')
ax.set_ylabel('Entropy')
ax.set_zlabel('num points')

ax.set_xlim3d(0, 1800)
ax.set_ylim3d(0.0, 1.0)
ax.set_zlim3d(0, 100) #750

verts = []
zs = u3[:,0]
facecolors =[]
for z in u3[:,0]:
    ux = u1[(u1[:,1]==z) & (u1[:,2]>20) ] #this predicate extracts entries only for this entropy value, second part of he predicate ensure we only consider eigenvectors that apply to more than 10 points
    xs = np.concatenate([[0],ux[:,0],[0]]) #eig
    ys = np.concatenate([[0],ux[:,2],[0]]) #num points
    #print xs,ys
    if z!=0: #this is the first entropy layer - equates to noise
        print(ux[:,0],ux[:,2])
    verts.append(list(zip(xs,ys)))
    facecolors.append((z, z * z, 0.0, 0.6))

poly = PolyCollection(verts, facecolors=facecolors)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')
plt.show()