import numpy as np
from laspy.file import File
import os
from sklearn.neighbors import NearestNeighbors

def corridor(c, conductor_condition, R=1, S=2):

	v1 = inFile.eig20
	v2 = inFile.eig21
	v3 = inFile.eig22 # these are the three components of eigenvector 2
	
	cRestrict =  c[:, conductor_condition]
	print(cRestrict.shape)
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(cRestrict))	
	distances, indices = nhbrs.kneighbors(np.transpose(c))
	nns = indices[:,0]
	v = np.vstack((v1,v2,v3))[:, conductor_condition][:,nns] #(3,N)
	u = c[:,:]-c[:, conductor_condition][:,nns]
	scale =(u[0,:]*v[0,:]+u[1,:]*v[1,:]+u[2,:]*v[2,:])
	
	w = u-scale*v
	w_norms = np.sqrt(w[0,:]**2+w[1,:]**2+w[2,:]**2)
	condition = (w_norms<R)&(np.absolute(scale)<S)
	
	return condition


for filename in os.listdir():
	if "Infty" in filename and filename[:4]=="attr":
		print(filename)
		inFile = File(filename, mode = "r")
		X = inFile.x
		Y = inFile.y
		Z = inFile.z
		c = np.vstack((X,Y,Z)) #(3,N)
		out = File("classified"+filename, mode = "w", header = inFile.header)
		out.points = inFile.points
		intensity = 0*inFile.intensity
		classification = inFile.classification
		classification[:] = 0
		stack = np.vstack((inFile.eig20, inFile.eig21, inFile.eig22))
		unq, ind, inv, cnt = np.unique(np.round(stack,2), return_index=True, return_inverse=True, return_counts=True, axis=1)
		conductor = ((inFile.eigent < 0.02) & (inFile.eig0 < 0.1) & (inFile.eig2 > 0.90) & (inFile.eig1 < 0.2) & (inFile.linearity > 0.9) & (inFile.ang2 > 0.1) & (inFile.return_num < inFile.num_returns) & (inFile.reader.get_dimension("1dist") < 0.3)) #& (cnt[inv] >= 10*inFile.optptdens))
		conductor = corridor(c, conductor, R = 1, S = 2)
		classification[conductor] = 1

		candidates1 = c[:, conductor]
		intensity1 = intensity[conductor]
		unq1, ind1, inv1, cnt1 = np.unique(np.round(candidates1/3,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		intensity1[:] = 0
		for item in np.arange(ind1.size):
			intensity1[inv1 == item] = 1000*np.median(inFile.scattering[conductor][inv1 == item])
		intensity[conductor] = intensity1
		classification[intensity > 3] = 0

		pylon =  (inFile.iso >= 0.5) & (inFile.iso < 0.75) & (inFile.ang2 < 0.1) & (inFile.eig0 < 0.25) & (inFile.linearity > 0.1) & (inFile.eigent > 0.2)
		pylon = corridor(c, pylon, R = 4, S = 1) & np.logical_not(conductor)

		candidates2 = c[:, pylon]
		intensity2 = intensity[pylon]
		unq2, ind2, inv2, cnt2 = np.unique(np.round(candidates2/3,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		intensity[:] = 0
		for item in np.arange(ind2.size):
			A = np.std(candidates2[:, inv2 == item], axis = 1)
			intensity2[inv2 == item] = A[2]/np.sqrt(A[0]**2+A[1]**2)
		intensity[pylon] = intensity2
		
		classification[pylon & np.logical_not(conductor)]=2
		classification[(intensity < 1)  & np.logical_not(conductor)] = 0

		nhbrsc = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c[:, classification == 1]))	
		distancesc, indicesc = nhbrsc.kneighbors(np.transpose(c[:,:]))
		in_corridor = distancesc[:,0] < 1
		classification[in_corridor & (classification == 0)] = 3

		out.intensity = intensity
		out.classification = classification
		out.close()
		print(filename, "done")
			
		

'''
Attributes
eig0
eig1
eig2
iso
eigent
scattering
linearity
planarity
entent
ang0
ang1
ang2
eig20
eig21
eig22
eig10
eig11
eig12
eig00
eig01
eig02
maxptdens
maxdist
1ptdens
1dist
optptdens
optdist
'''
