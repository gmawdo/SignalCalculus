import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time

timingstxt = open("CLASSN_TIMINGS.txt", "w+")

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

tile_condition = lambda x: "attr" in x and "classified" not in x

for tile_name in os.listdir():
	if tile_condition(tile_name):
		start = time.time()
		inFile = File(tile_name,mode = "r")
		Coords = np.stack((inFile.x,inFile.y,inFile.z))
		inFile = File(tile_name, mode = "r")
		LPS = np.stack((inFile.codim2, inFile.codim1, inFile.codim0), axis = 1)

		dims = np.argmax(LPS, axis = 1)+1
		dims[inFile.eig2<=0]=7

		if (dims == 1).any():
			v0 = 1*inFile.eig20
			v1 = 1*inFile.eig21
			v2 = 1*inFile.eig22
			condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
			v0[condition]=-v0[condition]
			v1[condition]=-v1[condition]
			v2[condition]=-v2[condition]
			v = np.vstack((5*v0,5*v1,5*v2, inFile.x, inFile.y, inFile.z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, dims == 1]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': inFile.x[dims == 1],
				'Y': inFile.y[dims == 1],
				'Z': inFile.z[dims == 1]
				}
			df = pd.DataFrame(frame)
			maxs = (df.groupby('A').max()).values
			mins = (df.groupby('A').min()).values
			unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
			lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
			lengths[labels==-1]=0
			classn = 0*inFile.classification
			classn1 = classn[dims == 1]
			classn1[:] = 1
			classn1[lengths<=2]=0
			classn[dims == 1]=classn1
			conductor = corridor(Coords, classn == 1, R=0.5, S=2)
			classn[conductor]=1
			classn[dims == 7] = 7
		
		prepylon = (dims == 2)&(classn != 1)
		if prepylon.any():
			v0 = 1*inFile.eig00
			v1 = 1*inFile.eig01
			v2 = 1*inFile.eig02
			condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
			v0[condition]=-v0[condition]
			v1[condition]=-v1[condition]
			v2[condition]=-v2[condition]
			v = np.vstack((5*v0,5*v1,5*v2, inFile.x, inFile.y, inFile.z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, prepylon]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': inFile.x[prepylon],
				'Y': inFile.y[prepylon],
				'Z': inFile.z[prepylon]
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
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 2]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords))
			classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) ]=2

		preveg = (dims == 3) & (classn != 2) & (classn != 1)
		if preveg.any():
			v = np.vstack((inFile.x, inFile.y, inFile.z))
			clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, preveg]).transpose(1,0))
			labels = clustering.labels_
			frame =	{
				'A': labels,
				'X': inFile.x[preveg],
				'Y': inFile.y[preveg],
				'Z': inFile.z[preveg]
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
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 3]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords))
			classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) & (classn != 2)]=3

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
		outFile.classification = classn
		outFile.close()
		print(tile_name)

		end = time.time()
		timingstxt.write(tile_name[:-4]+" "+str(end-start)+"\n")

timingstxt.close()
