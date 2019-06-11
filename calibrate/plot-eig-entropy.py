import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File
from sklearn.neighbors import NearestNeighbors

def knockoutOutlyers(classification,writecondition,delta):
    #this method takes a condition as the basis for a subset of points and dismiesses outlying points - ones with distant neighbours
    #distant determined by input parm
    c = np.vstack((inFile.x,inFile.y,inFile.z)) #(3,N)

    conductor_class = classification[writecondition]
    cRestrict =  c[:, writecondition]    
    if cRestrict.size > 0:
        nhbrs = NearestNeighbors(n_neighbors = 2, algorithm = "kd_tree").fit(np.transpose(cRestrict))   
        distances, indices = nhbrs.kneighbors(np.transpose(cRestrict))
        conductor_class[(distances[:,1] > delta)]=0
        classification[writecondition] = conductor_class
        return classification


inFile = File("ENEL/000/attrDF2000305_Completa.laz.las-GpsTime139236.86195908333333333332139256.48610712499999999998N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")

ptCondition=(inFile.X==58805768)*(inFile.Y==507559564)*(inFile.Z==30717)
print(inFile.impdec[ptCondition],inFile.cart2sph[ptCondition],inFile.ent[ptCondition],inFile.iso[ptCondition],inFile.eig1[ptCondition],inFile.eig2[ptCondition],inFile.eig0[ptCondition],inFile.rank[ptCondition])
ptCondition2=(inFile.X==58804866)*(inFile.Y==507561809)*(inFile.Z==27814)
print(inFile.impdec[ptCondition2],inFile.cart2sph[ptCondition2],inFile.ent[ptCondition2],inFile.iso[ptCondition2],inFile.eig1[ptCondition2],inFile.eig2[ptCondition2],inFile.eig0[ptCondition2],inFile.rank[ptCondition2])
print(inFile.x[ptCondition2])

ptCondition3=(inFile.X==58795450)*(inFile.Y==507542192)*(inFile.Z==30993)
print(inFile.impdec[ptCondition3],inFile.cart2sph[ptCondition3],inFile.ent[ptCondition3],inFile.iso[ptCondition3],inFile.eig1[ptCondition3],inFile.eig2[ptCondition3],inFile.eig0[ptCondition3],inFile.rank[ptCondition3])
print(inFile.x[ptCondition3])

# create matrix of all the eigenvectors
result1 = np.stack((inFile.cart2sph,inFile.ent), axis=-1)
print("formed main result set size:",len(result1))

# find how many unique eigenvectors in combo with entropy
u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
c1 = c1.reshape(u1.shape[0],1)
u1 = np.concatenate((u1,c1),axis=1)

#PREDICATES!!!!!!!
#entropy_range = (u1[:,1]>=0)*(u1[:,1]<0.02)*(u1[:,0]>100)*(u1[:,0]<200)
entropy_range = (u1[:,1]>0)*(u1[:,1]<0.02)
pylon_predicate = (inFile.iso >= 0.6) & (inFile.iso < 0.75) & (inFile.lang < 0.1)

u1 = u1[entropy_range]
u1.sort

u3,i3,c3 = np.unique(u1[:,[1]], axis=0,return_inverse=True,return_counts=True) #ent
print("number of distinct entropies:",u3.shape)

#classify a set of points into a new file
outFile = File("Gary.las", mode = "w", header = inFile.header)
classification = inFile.classification
outFile.points = inFile.points
outFile.intensity = inFile.ent*1.0e16

classification*=0

#classi = 11
#for s in u3[:,0]: #for each entropy
#    writecondition = (np.round(inFile.ent,3)==s) #*(inFile.lang>0.45)*(inFile.iso>0.55)*(inFile.iso<0.65)
#    classification[writecondition]=classi
#    #classi = classi + 1 #uncomment this line if we want a different class for each entropy layer

#classification=1+(10*(np.round(inFile.eig0,2)==0))+(10*(np.round(inFile.eig1,2)==0))+(10*(np.round(inFile.eig2,2)==0))

basecondition = (inFile.ent>0.002)*(inFile.ent<0.02)#*(inFile.iso>0.55)*(inFile.iso<0.65)

conductor = (1+(10*(np.round(inFile.eig0,2)==0))+(10*(np.round(inFile.eig1,2)==0))+(10*(np.round(inFile.eig2,2)==0))==11)
#classification[conductor] = 11
conductor = (inFile.impdec==1000)*(inFile.ent>0.000000002)*(1+(10*(np.round(inFile.eig0,2)==0))+(10*(np.round(inFile.eig1,2)==0))+(10*(np.round(inFile.eig2,2)==0))==21)
classification[conductor] = 21
conductor = (inFile.impdec==1000)*(inFile.ent==0)*(inFile.iso==0)*(1+(10*(np.round(inFile.eig0,2)==0))+(10*(np.round(inFile.eig1,2)==0))+(10*(np.round(inFile.eig2,2)==0))==31)
classification[conductor] = 31



#classify verticals
classification[pylon_predicate]=8

#for s in u3[:,0]: #for each entropy
#    classification = knockoutOutlyers(classification,(np.round(inFile.ent,3)==s)*(classification>=11),3)
#classification = knockoutOutlyers(classification,(classification==11),3)
#classification = knockoutOutlyers(classification,(classification==8),0.25)

outFile.classification = classification
outFile.close()

inFile.close()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.view_init(elev=16, azim=43)
#
#ax.set_xlabel('eigenvector')
#ax.set_ylabel('Entropy')
#ax.set_zlabel('num points')
#
#ax.set_xlim3d(0, 1800)
#ax.set_ylim3d(0.0, 1.0)
#ax.set_zlim3d(0, 100) #750
#
#verts = []
#zs = u3[:,0]
#facecolors =[]
#for z in u3[:,0]:
#    ux = u1[(u1[:,1]==z) & (u1[:,2]>10) ] #this predicate extracts entries only for this entropy value, second part of he predicate ensure we only consider eigenvectors that apply to more than 10 points
#    xs = np.concatenate([[0],ux[:,0],[0]]) #eig
#    ys = np.concatenate([[0],ux[:,2],[0]]) #num points
#    #print xs,ys
#    #if z!=0: #this is the first entropy layer - equates to noise
#        #print(ux[:,0],ux[:,2])
#    verts.append(list(zip(xs,ys)))
#    facecolors.append((z, z * z, 0.0, 0.6))
#
#poly = PolyCollection(verts, facecolors=facecolors)
#poly.set_alpha(0.7)
#ax.add_collection3d(poly, zs=zs, zdir='y')
#plt.show()#