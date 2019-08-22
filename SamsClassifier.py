import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time



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

class Eigenvector:
	def __init__(self, vector):
		self.vector = vector
	def __eq__(self, other):
		p = np.dot(self.vector,self.vector)-2*np.dot(self.vector,other.vector)+np.dot(other.vector,other.vector)
		return np.round(int(100*(1-np.absolute(np.cos(1-0.5*p)))))==0

def listunq(L):
	output = []
	inverse = []
	first_index = 0
	le = len(L)
	for index, x in enumerate(L):
		if x not in output:
			print(index*100/le)
			output.append(x)
			inverse.append(first_index)
		first_index += 1
	unq, ind, inv, cnt = np.unique(np.array(inverse), return_index=True, return_inverse=True, return_counts=True)
	return ind, inv, cnt

def dimension_count(dimensionality_array):
	#A = np.array([[1/3, 1/3, 1/3], [1 , 0, 0], [0,1,0], [0,0,1], [0.5, 0, 0.5], [0,0.5,0.5], [0.5, 0.5, 0]]) # (7,3)
	A = np.array([[1 , 0, 0], [0,1,0], [0,0,1]])
	C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], dimensionality_array[None,:, :])), axis = -1) #(7, M1, 3, 2)
	JSD = jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
	dims = np.argmin(JSD, axis = 0)
	return dims

i=0
for file_name in os.listdir():
	if file_name[-4:]==".las":
		start = time.time()
		inFile = File(file_name,mode = "rw")
		Coords = np.stack((inFile.x,inFile.y,inFile.z))
		M = len(inFile)
		x = np.sqrt(inFile.eig2)
		y = np.sqrt(inFile.eig1)
		z = np.sqrt(inFile.eig0)
		l = np.minimum(x, 1-y)
		p = np.minimum(y, 1-z)
		s = z
		L = l/(l+p+s)
		P = p/(l+p+s)
		S = s/(l+p+s)
		B = np.stack((L,P,S), axis = -1) #(M,3)
		dims = dimension_count(B)
		dims[inFile.eig2<=0]=7
		DIMS = 1*dims
		
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, dims ==2]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, dims == 1]))
		dims1 = dims[dims==1]
		dims1[distances[:,-1]<2]=2
		dims[dims==1]=dims1

		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, dims ==2]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, dims == 0]))
		dims0 = dims[dims==0]
		dims0[distances[:,-1]<0.5]=2
		dims[dims==0]=dims0

		clustering = DBSCAN(eps=0.5, min_samples=1).fit((Coords[:, dims == 0]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
					'A': labels,
					'X': 1*(inFile.num_returns[dims==0]==inFile.return_num[dims==0]),
					'Y': np.ones((labels.size,))
					}
		df = pd.DataFrame(frame)
		sums = (df.groupby('A').sum()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		props = (sums[:,0]/sums[:,1])[inv]
		props[labels==-1]=1
		dims0= dims[dims == 0]
		dims0[props>0.33]=2
		dims[dims==0]=dims0

		v0 = inFile.eig20
		v1 = inFile.eig21
		v2 = inFile.eig22
		condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
		v0[condition]=-v0[condition]
		v1[condition]=-v1[condition]
		v2[condition]=-v2[condition]
		v = np.vstack((5*v0,5*v1,5*v2, inFile.x, inFile.y, inFile.z))


		clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, dims == 0]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
			'A': labels,
			'X': inFile.x[dims==0],
			'Y': inFile.y[dims==0],
			}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn = 0*inFile.classification
		classn1 = classn[dims == 0]
		classn1[:] = 1
		classn1[lengths<=2]=0
		classn[dims == 0]=classn1

		if (classn==1).any():
			conductor = corridor(Coords, classn==1, R=1, S=2)
			classn[:]=0
			classn[conductor] = 1

			prepylon = (inFile.ang2<0.2) & (dims == 1) & (classn!=1)
			if prepylon.any():
				nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:2, classn == 1]))
				distances, indices = nhbrs.kneighbors(np.transpose(Coords[:2, prepylon]))
				prepylon_res = prepylon[prepylon]
				prepylon_res[distances[:,0] > 1]=False
				if prepylon_res.any():
					prepylon[prepylon]=prepylon_res
					pylon = corridor(Coords, prepylon, R=1, S=4)
					classn[pylon & (classn!=1)]=2

			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:3, classn == 1]))
			K = (inFile.diment>0.2)&(inFile.return_num<inFile.num_returns)&(classn!=1)&(classn!=2)
			distances, indices = nhbrs.kneighbors(np.transpose(Coords[:3, K]))
			pdist = distances[:,0]
			classn2 = classn[K]
			classn2[:] = 4
			classn2[pdist > 0.5] = 0
			classn[K] = classn2

			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:3, dims == 1]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords[:3, K]))

		inFile.classification = classn
		inFile.intensity = 0*inFile.intensity
		inFile.close()
		print(file_name)
		i+=1
		





		

		
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
