import numpy as np
from laspy.file import File
from sklearn.neighbors import NearestNeighbors

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

def corridor(conductor_condition):
	outFile = File("Corridor.las", mode = "w", header = inFile.header)
	outFile.points = inFile.points
	v1 = inFile.lambda_x
	v2 = inFile.lambda_y
	v3 = inFile.lambda_z # these are the three components of eigenvector 2
	classification = inFile.classification
	classification = 0*classification
	X = inFile.x
	Y = inFile.y
	Z = inFile.z
	c = np.vstack((X,Y,Z)) #(3,N)
	

	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c[:, conductor_condition]))		
	distances, indices = nhbrs.kneighbors(np.transpose(c))
	nns = indices[:,0]
	v = np.vstack((v1,v2,v3))[:, conductor_condition][:,nns] #(3,N)
	u = c[:,:]-c[:, conductor_condition][:,nns]
	scale =(u[0,:]*v[0,:]+u[1,:]*v[1,:]+u[2,:]*v[2,:])
	
	w = u-scale*v
	w_norms = np.sqrt(w[0,:]**2+w[1,:]**2+w[2,:]**2)
	condition = (w_norms<1)&(np.absolute(scale)<2)

	classification[condition] = 2
	outFile.intensity = 100*np.sqrt(u[0,:]**2+u[1,:]**2+u[2,:]**2)
	outFile.classification = classification
	outFile.close()

	#Eigenvector QC
	outFile1 = File("EigQC.las", mode = "w", header = inFile.header)
	outFile1.points = inFile.points[distances[:,0]<1]
	outFile1.x = 50*v1[distances[:,0]<1]+np.mean(inFile.x)
	outFile1.y = 50*v2[distances[:,0]<1]+np.mean(inFile.y)
	outFile1.z = 50*v3[distances[:,0]<1]+np.mean(inFile.z)
	outFile1.classification = classification[distances[:,0]<1]
	outFile1.close()
	


corridor((0.002<inFile.ent)*(0<inFile.iso)*(inFile.ent<0.02)*(30<COUNTS[:,0]))

