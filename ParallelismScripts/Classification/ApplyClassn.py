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


def corridor(c, d, conductor_condition, R=1, S=2):

	if d == 4:
		v0 = 1*inFile.eig31[IND]
		v1 = 1*inFile.eig32[IND]
		v2 = 1*inFile.eig33[IND]
	if d == 3:
		v0 = 1*inFile.eig20[IND]
		v1 = 1*inFile.eig21[IND]
		v2 = 1*inFile.eig22[IND]
	cRestrict =  c[:, conductor_condition]
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(cRestrict))
	distances, indices = nhbrs.kneighbors(np.transpose(c))
	nns = indices[:,0]
	v = np.vstack((v0,v1,v2))[:, conductor_condition][:,nns] #(3,N)
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
	A = np.array([[1 , 0, 0], [0,1,0], [0,0,1]])
	C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], dimensionality_array[None,:, :])), axis = -1) #(7, M1, 3, 2)
	JSD = jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
	dims = np.argmin(JSD, axis = 0)
	return dims

def tile_condition(tile_name):
	return "attr" in tile_name and "classified" not in tile_name

for tile_name in os.listdir():
	if tile_condition(tile_name):
		start = time.time()
		inFile = File(tile_name,mode = "r")
		voxel = inFile.vox
		UNQ, IND, INV, CNT = np.unique(voxel, return_index=True, return_inverse=True, return_counts=True)
		if (inFile.vox == 0).all():
			IND = np.arange(len(inFile))
			INV = IND
		u = 0.0*IND+0.05
		x = inFile.x[IND]
		y = inFile.y[IND]
		z = inFile.z[IND]
		dim1 = inFile.dim1[IND]
		dim2 = inFile.dim2[IND]
		dim3 = inFile.dim3[IND]
		eig1 = inFile.eig1[IND]
		eig2 = inFile.eig2[IND]
		eig0 = inFile.eig0[IND]
		classification = inFile.classification[IND]
		Coords = u[None,:]*np.floor(np.stack((x/u,y/u,z/u), axis = 0))
		LPS = np.stack((dim1, dim2, dim3), axis = 1)
		dims = 1+np.argmax(LPS, axis = 1)
		classn = 0*classification
		try:
			dim4 = inFile.dim4[IND]
			d = 4
		except:
			d = 3
		
		if d==4:
			eig3 = inFile.eig3

		if d == 3:
			dims[eig2<=0]=7
		if d == 4:
			dims[eig3<=0]=7

		if (dims == 1).any():
			if d == 4:
				v0 = 1*inFile.eig31[IND]
				v1 = 1*inFile.eig32[IND]
				v2 = 1*inFile.eig33[IND]
			if d == 3:
				v0 = 1*inFile.eig20[IND]
				v1 = 1*inFile.eig21[IND]
				v2 = 1*inFile.eig22[IND]
			condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
			v0[condition]=-v0[condition]
			v1[condition]=-v1[condition]
			v2[condition]=-v2[condition]
			v = np.vstack((5*v0,5*v1,5*v2, x, y, z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, dims == 1]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': x[dims == 1],
				'Y': y[dims == 1],
				'Z': z[dims == 1]
				}
			df = pd.DataFrame(frame)
			maxs = (df.groupby('A').max()).values
			mins = (df.groupby('A').min()).values
			unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
			lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
			lengths[labels==-1]=0
			classn1 = classn[dims == 1]
			classn1[:] = 1
			classn1[lengths<=2]=0
			classn[dims == 1]=classn1
			if (classn == 1).any():
				conductor = corridor(Coords, d, classn == 1, R=0.5, S=2)
				classn[conductor]=1
			classn[dims == 7] = 7
		
		prepylon = (dims == 2)&(classn != 1)
		if prepylon.any():
			if d == 4:
				v0 = 1*inFile.eig21[IND]
				v1 = 1*inFile.eig22[IND]
				v2 = 1*inFile.eig23[IND]
			if d == 3:
				v0 = 1*inFile.eig10[IND]
				v1 = 1*inFile.eig11[IND]
				v2 = 1*inFile.eig12[IND]
			condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
			v0[condition]=-v0[condition]
			v1[condition]=-v1[condition]
			v2[condition]=-v2[condition]
			v = np.vstack((5*v0,5*v1,5*v2, x, y, z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, prepylon]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': x[prepylon],
				'Y': y[prepylon],
				'Z': z[prepylon],
				}
			df = pd.DataFrame(frame)
			maxs = (df.groupby('A').max()).values
			mins = (df.groupby('A').min()).values
			unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
			lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
			lengths[labels==-1]=0
			classn2 = classn[prepylon]
			classn2[:] = 2
			classn2[lengths<=2]=0
			classn[prepylon]=classn2
			if (classn == 2).any():
				nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 2]))
				distances, indices = nhbrs.kneighbors(np.transpose(Coords))
				classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) ]=2

		preveg = (dims == 3) & (classn != 2) & (classn != 1)
		if preveg.any():
			v = np.vstack((x, y, z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, preveg]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': x[preveg],
				'Y': y[preveg],
				'Z': z[preveg]
				}
			df = pd.DataFrame(frame)
			maxs = (df.groupby('A').max()).values
			mins = (df.groupby('A').min()).values
			unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
			lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
			lengths[labels==-1]=0
			classn3 = classn[preveg]
			classn3[:] = 3
			classn3[lengths<2]=0
			classn[preveg] = classn3
			if (classn == 3).any():
				nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 3]))
				distances, indices = nhbrs.kneighbors(np.transpose(Coords))
				classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) & (classn != 2)]=3
			
		if ((classn != 0) & (classn != 7)).any():
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, (classn != 0) & (classn != 7)]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, classn == 0]))
			classn0 = classn[classn == 0]
			classn0[(distances[:,0]<0.5)] = (classn[(classn != 0) & (classn != 7)])[indices[(distances[:,0]<0.5),0]]
			classn[(classn==0)] = classn0

		if (classn == 1).any() and (classn == 3).any():
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 1]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, classn == 3]))
			classn3 = classn[classn==3]
			classn3[distances[:,0]<0.5]=4
			classn[classn==3]=classn3

		outFile = File("classified"+tile_name, mode = "w", header = inFile.header)
		outFile.points = inFile.points
		outFile.classification = classn[INV]
		outFile.close()
		print(tile_name)

		end = time.time()
		timingstxt.write(tile_name[:-4]+" "+str(end-start)+"\n")

timingstxt.close()
