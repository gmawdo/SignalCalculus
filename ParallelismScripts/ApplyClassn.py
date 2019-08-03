import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time

timingstxt = open("CLASSN_TIMINGS.txt", "w+")

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


def dimension_count(dimensionality_array):
	A = np.array([[1/3, 1/3, 1/3], [1 , 0, 0], [0,1,0], [0,0,1], [0.5, 0, 0.5], [0,0.5,0.5], [0.5, 0.5, 0]]) # (7,3)
	C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], dimensionality_array[None,:, :])), axis = -1) #(7, M1, 3, 2)
	JSD = jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
	dims = np.argmin(JSD, axis = 0)
	return dims

def condition(file_name):
	return (file_name[-4:]==".las") and ("attr" in file_name) and ("NFL" in file_name) and not("classified" in file_name)

if not("OUTPUTS" in os.listdir()):
	os.system("mkdir OUTPUTS")

for tile_name in os.listdir():
	if condition(tile_name):
		start = time.time()

		inFile = File(tile_name)
		Coords = np.vstack((inFile.x, inFile.y, inFile.z)) #(3,M)
		M = len(inFile)
		B = np.stack((inFile.linearity, inFile.planarity, inFile.scattering), axis = -1) #(M,3)
		dims = dimension_count(B)
		dims[inFile.reader.get_dimension("onedist")>0.5]=7
		dims_save = 1*dims
		
		unq, ind, inv, cnt = np.unique(np.round(Coords[:3,:]/2,0), return_index=True, return_inverse=True, return_counts=True, axis=1)

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
		
		# set up entropies for later use and for alternative classification of conductors
		entropies = entropy(X/(np.sum(X, axis = 1)[:,None]))[inv]
		c = {arg: 1000*X[inv, arg] for arg in range(7)}
		conditions = 	[
						c[0] <= 10*cnt[inv],
						c[1] >= 800*cnt[inv],
						c[2] <= 100*cnt[inv],
						c[3] <= 1*cnt[inv],
						c[4] <= 1*cnt[inv],
						c[5] <= 1*cnt[inv],
						10*c[6] <= c[1],
						]
		
		classn = 1*dims
		classn[:] = 1
		for item in conditions:
			classn[np.logical_not(item)] = 0
		classn[inFile.ang2<0.5] = 0

		dim1 = classn == 1


		clustering = DBSCAN(eps=0.5, min_samples=1).fit((Coords[:, dim1]).transpose(1,0))
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

		clustering = DBSCAN(eps=0.5, min_samples=1).fit((Coords[:, classn==1]).transpose(1,0))
		labels = clustering.labels_
		
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
		classn1[lengths<=1]=0
		classn[classn==1]=classn1

		boolean_conductor = corridor(Coords, classn == 1, R=1, S=2)
		classn[:] = 0
		pylon = (np.sum(X[:, [1,2,6]], axis = 1)/np.sum(X, axis = 1))[inv_save] > 0.9
		classn[pylon & (inFile.ang2 < 0.2)] = 2
		classn[boolean_conductor] = 1

		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:3, classn == 1]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:3, classn == 2]))
		pdist = distances[:,0]
		classn2 = classn[classn == 2]
		classn2[pdist > 1] = 0
		classn[classn == 2] = classn2

		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:3, classn == 2]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:3, classn != 1]))
		classnnot1 = classn[classn!=1]
		classnnot1[distances[:,0] > 1] = 0

		Pylon = corridor(Coords, classn == 2, 1, 4) 
		classn[classn != 1] =0
		classn[Pylon & np.logical_not(classn == 1)] = 2

		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 1]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, classn == 0]))
		classn0 = classn[classn == 0]
		classn0[distances[:,0] <= 0.5] = 4
		classn[classn == 0] = classn0
		classn[(classn == 4) & (entropies < 0.7)] = 0

		os.chdir("OUTPUTS")
		out = File("classified"+tile_name, mode = "w", header = inFile.header)
		out.points = inFile.points
		out.classification = classn
		out.close()
		os.chdir("..")

		end = time.time()
		timingstxt.write(tile_name[:-4]+" "+str(end-start)+"\n")

timingstxt.close()
