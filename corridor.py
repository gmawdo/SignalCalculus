import numpy as np
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import time

start = time.time() 

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


inFile = File("OLD-TEST-FILES/attrTile9NFLClip100_00N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")


print("file read")
# inFile.x
# create matrix of all the eigenvectors
result1 = np.stack((inFile.cart2sph,np.round(inFile.ent,3)), axis=-1)
print("formed main result set size:",len(result1))

# find how many unique eigenvectors in combo with entropy
u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
c1 = c1.reshape(u1.shape[0],1)
u1 = np.concatenate((u1,c1),axis=1)

iso_condition = (0.5<inFile.iso)&(inFile.iso<0.6)
lang_condition = inFile.lang>0.4
UID = u1[i1,0]
COUNTS = c1[i1]
X = inFile.x
Y = inFile.y
Z = inFile.z



outFile = File("Corridor.las", mode = "w", header = inFile.header)
outFile.points = inFile.points
classification = inFile.classification
classification = 0*classification
def corridor(conductor_condition, R=1, S=2):

	v1 = inFile.lambda_x
	v2 = inFile.lambda_y
	v3 = inFile.lambda_z # these are the three components of eigenvector 2

	c = np.vstack((X,Y,Z)) #(3,N)
	

	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c[:, conductor_condition]))		
	distances, indices = nhbrs.kneighbors(np.transpose(c))
	nns = indices[:,0]
	v = np.vstack((v1,v2,v3))[:, conductor_condition][:,nns] #(3,N)
	u = c[:,:]-c[:, conductor_condition][:,nns]
	scale =(u[0,:]*v[0,:]+u[1,:]*v[1,:]+u[2,:]*v[2,:])
	
	w = u-scale*v
	w_norms = np.sqrt(w[0,:]**2+w[1,:]**2+w[2,:]**2)
	condition = (w_norms<R)&(np.absolute(scale)<S)
	
	return condition

	
conductor = corridor((0.002<inFile.ent)*(0<inFile.iso)*(inFile.ent<0.02)*(30<COUNTS[:,0]))
classification[conductor] = 1


c2d = np.vstack((X,Y))

nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c2d[:, classification == 1]))		
distances2d, indices2d = nhbrs2d.kneighbors(np.transpose(c2d))
classification[distances2d[:,0]>1]=0


c3d = np.vstack((X,Y,Z))
nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c3d[:, (classification == 1)]))		
distances3d, indices3d = nhbrs2d.kneighbors(np.transpose(c3d))

pylon_condition = (0.5>=inFile.lang)&(0.2>=inFile.curv)&(inFile.rank ==3)&(inFile.plan_reg<0.8)&(inFile.plang>=0.5)&(inFile.ent<0.7)&(distances3d[:,0]<1)
classification[pylon_condition]=2

classification[(distances3d[:,0]<1)&(classification!=1)&(classification!=2)]=3
classification[distances2d[:,0]>1]=0

outFile.classification = classification

outFile.intensity = intensity

end = time.time()
print("done. Time taken = "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

outFile.close()





