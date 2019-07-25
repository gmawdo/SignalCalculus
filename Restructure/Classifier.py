import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import lasmaster as lm
import matplotlib.pyplot as plt

os.chdir("testingLAS")

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


for file_name in os.listdir():	
	if file_name[:4]=="attr" and not("jsd" in file_name) and ("TILE19" in file_name):
		inFile = File(file_name)
		M = len(inFile)
		B = np.stack((inFile.linearity, inFile.planarity, inFile.scattering), axis = -1) #(M,3)
		Coords = np.vstack((inFile.x, inFile.y, inFile.z)) #(3,M)
		unq, ind, inv, cnt = np.unique(np.round(Coords[:3,:]/6,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		os.chdir("OUTPUTS")
		C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], B[None,:, :])), axis = -1) #(7, M1, 3, 2)
		JSD = jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
		dims = np.argmin(JSD, axis = 0)
		dims[inFile.reader.get_dimension("1dist")>0.5]=7

		frame = {'A': inv,
			str(0): (dims==0).astype(int),
			str(1): (dims==1).astype(int),
			str(2): (dims==2).astype(int),
			str(3): (dims==3).astype(int),
			str(4): (dims==4).astype(int),
			str(5): (dims==5).astype(int),
			str(6): (dims==6).astype(int),
			}
		df = pd.DataFrame(frame)
		X = (df.groupby('A').sum()).values #(M1,3)
		inv_save = inv
		
		entropies = entropy(X/(np.sum(X, axis = 1)[:,None]))[inv]

		#the below are dangerous -- we need a better QC
		c = {arg: 1000*X[inv, arg] for arg in range(7)}
		conditions = 	[
						c[0] <= 10*cnt[inv],
						c[1] >= 800*cnt[inv],
						c[2] <= 100*cnt[inv],
						c[3] <= 1*cnt[inv],
						c[4] <= 1*cnt[inv],
						c[5] <= 1*cnt[inv],
						c[6] <= 10*cnt[inv],
						]
		classn = 1*dims
		classn[:] = 1
		for item in conditions:
			classn[np.logical_not(item)] = 0
		dim1 = classn == 1


		clustering = DBSCAN(eps=0.5, min_samples=4).fit((Coords[:, dim1]).transpose(1,0))
		labels = clustering.labels_

		frame =	{
					'A': labels,
					'X': inFile.x[dim1],
					'Y': inFile.y[dim1],
					}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn1 = classn[classn == 1]
		classn1[lengths<=1]=0
		classn[classn==1]=classn1

		clustering = DBSCAN(eps=1.25, min_samples=8).fit((Coords[:, classn==1]).transpose(1,0))
		labels = clustering.labels_
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)

		dim1 = classn == 1

		frame =	{
					'A': labels,
					'X': inFile.x[dim1],
					'Y': inFile.y[dim1],
					}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn1 = classn[dim1]
		classn1[lengths<=3]=0
		classn[classn==1]=classn1

		boolean_conductor = corridor(Coords, classn == 1, R=1, S=2)
		classn[:] = 0
		pylon = (np.sum(X[:,[1,2,6]], axis = 1)/np.sum(X, axis = 1))[inv_save] >0.9
		classn[pylon & (inFile.ang2 < 0.2)] = 2
		classn[boolean_conductor] = 1

		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:2, classn == 1]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:2, classn == 2]))
		z = inFile.z
		zdist = z[classn == 2]-(z[classn==1][indices[:,0]])
		classn2 = classn[classn == 2]
		classn2[(distances[:,0]>0.5) | (zdist > 1)] = 0
		clustering = DBSCAN(eps=1, min_samples=4).fit((Coords[:, classn==2]).transpose(1,0))
		labels = clustering.labels_
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		classn2[labels == -1] = 0


		classn[classn == 2] = classn2
		Pylon = corridor(Coords, classn == 2, 1, 4) 
		classn[classn != 1] =0
		classn[Pylon & np.logical_not(classn == 1)] = 2
		
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 1]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, classn == 0]))
		classn0 = classn[classn == 0]
		classn0[distances[:,0] <= 0.5] = 4
		classn[classn == 0] = classn0
		classn[(classn == 4) & (entropies < 0.7)] = 0

		out = File("Conductor"+file_name, mode = "w", header = inFile.header)
		out.points = inFile.points

	
		out.classification = classn
		out.close()
		os.chdir("..")
		print("Conductor",file_name, "done")

		
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
