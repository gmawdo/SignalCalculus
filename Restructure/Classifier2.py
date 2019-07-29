import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import lasmaster as lm
import matplotlib.pyplot as plt

os.chdir("TILES_2017_04_20-07_52_14_3")

def jsd(distribution):
	M = distribution.shape[-2]
	N = distribution.shape[-1]
	return (entropy(np.mean(distribution, axis = -2))-np.mean(entropy(distribution), axis = -1))*np.log(N)/np.log(M)


def corridor(c, conductor_condition, R=1, S=2):

	v1 = inFile.eig20
	v2 = inFile.eig21
	v3 = inFile.eig22 # these are the three components of eigenvector 2
	cRestrict =  c[:, conductor_condition]
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

def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.sum(-distribution*logs, axis = -1)/np.log(N)
	return entropies

A = np.array([[1/3, 1/3, 1/3], [1,0,0], [0,1,0], [0,0,1], [0.5, 0, 0.5], [0,0.5,0.5], [0.5, 0.5, 0]]) # (7,3)

def dimensions(L,P,S):
	B = np.stack((L,P,S), axis = -1) #(M,3)
	C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], B[None,:, :])), axis = -1)
	JSD = jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
	dims = np.argmin(JSD, axis = 0)
	return dims

def X_array(voxel_ids, dims):
	frame = {'A': inv,
	str(0): ((dims==0) & (dims<=6)).astype(int),
	str(1): (dims==1).astype(int),
	}
		
	X = (pd.DataFrame(frame).groupby('A').sum()).values #(M1,3)
	return X

for file_name in os.listdir():	
	if "TILE11" in file_name and "attr" in file_name:
		inFile = File(file_name)
		classn = inFile.classification
		Coords = np.vstack((inFile.x,inFile.y,inFile.z)) #(3,M)
		unq, ind, inv, cnt = np.unique(np.round(Coords[:3,:]/0.5,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		os.chdir("OUTPUTS")

		dims = dimensions(inFile.linearity,inFile.planarity,inFile.scattering)
		dims[inFile.onedist>0.5]=7

		classn[:] = 0
		classn[dims == 1] = 1
		classn[inFile.entent > 0.1] = 0

		clustering = DBSCAN(eps = 0.5, min_samples=4).fit((Coords[:, classn == 1]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
					'A': labels,
					'X': inFile.x[classn == 1],
					'Y': inFile.y[classn == 1],
					}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn1 = classn[classn == 1]
		classn1[lengths<=1]=0
		classn[classn == 1]=classn1

		clustering = DBSCAN(eps=0.5, min_samples=4).fit((Coords[:, classn == 1]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
					'A': labels,
					'X': inFile.x[classn == 1],
					'Y': inFile.y[classn == 1],
					}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn1 = classn[classn == 1]
		classn1[lengths<=5]=0
		#classn[classn == 1]=classn1

		ncond = classn!=1
		classn[classn == 1] = labels
		classn += 1
		classn[(classn%32==0) & (classn!= 0)]+=1
		classn = classn%32
		classn[ncond] = 0

		out = File("Classified2"+file_name, mode = "w", header = inFile.header)
		out.points = inFile.points
		out.classification = classn
		out.intensity = dims
		out.close()
		os.chdir("..")
		print("Classified",file_name, "done")

		
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
oneptdens
onedist
optptdens
optdist
'''
