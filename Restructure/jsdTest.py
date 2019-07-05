import numpy as np
from laspy.file import File
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import lasmaster as lm

os.chdir("testingLAS")

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

A = np.array([[1/3, 1/3, 1/3], [1,0,0], [0,1,0], [0,0,1], [0.5, 0, 0.5], [0,0.5,0.5], [0.5, 0.5, 0]]) # (7,3)

for file_name in os.listdir():	
	if "Infty" in file_name and file_name[:4]=="attr" and "TILE22" in file_name and not("jsd" in file_name):
		inFile = File(file_name)
		M = len(inFile)
		P = np.stack((inFile.linearity, inFile.planarity, inFile.scattering), axis = -1) #(M,3)
		Coords = np.vstack((inFile.x, inFile.y, inFile.z)) #(M,3)
		unq, ind, inv, cnt = np.unique(np.round(Coords[:3,:]/2,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		M1 = ind.size
		os.chdir("jsd")
		df = pd.DataFrame({'A': inv, 'L': inFile.linearity, 'P': inFile.planarity, 'S': inFile.scattering,'I': inFile.optptdens, 'Z': inFile.z})
		X = (df.groupby('A').mean()).values #(M1,3) 
		B = X[:, :3]
		C = np.stack(tuple(np.broadcast_arrays(A[:,None,:], B[None,:, :])), axis = -1) #(7, M1, 3, 2)
		JSD = lm.infotheory.jsd(C.transpose(0, 1, 3, 2)) #(7, M1)
		dims = np.argmin(JSD, axis = 0)
		classn = dims[inv]
		classn[inFile.reader.get_dimension("1dist")>0.5]=7

		unq, ind, inv, cnt = np.unique(np.round(Coords[:2,:]/2,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		df['A'] = inv
		ground = (pd.DataFrame({'A':df['A'], 'Z':df['Z']}).groupby('A').quantile(0.01)).values
		df['Z'] = ground[df['A'],0]
		unq, ind, inv, cnt = np.unique(np.round(Coords[:2,:]/2,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		df['A'] = inv
		ground = (pd.DataFrame({'A':df['A'], 'Z':df['Z']}).groupby('A').quantile(0.99)).values
		df['Z'] = ground[df['A'],0]
		#unq, ind, inv, cnt = np.unique(np.round(Coords[:2,:]/2,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
		#ground = (pd.DataFrame({'A':inv, 'Z':df['Z']}).groupby('A').quantile(0.01)).values
		#df['Z'] = ground[inv,0]
		#ground = (pd.DataFrame({'A':inv, 'Z':df['Z']}).groupby('A').quantile(0.99)).values

		theta = np.linspace(0, np.pi, 1000)[:-1] # this represents the interval [0, pi)
		r = np.zeros(theta.shape, dtype = float)
		total = Coords[:, classn == 1].shape[1]
		pointids, thetaids = np.broadcast_arrays(np.arange(total)[:,None], np.arange(theta.size)[None,:])
		a, costheta = np.broadcast_arrays(Coords[0, classn == 1][:,None], np.cos(theta)[None,:])
		b, sintheta = np.broadcast_arrays(Coords[1, classn == 1][:,None], np.sin(theta)[None,:])
		r = a*costheta+b*sintheta
		R = (np.round(100*r,2)).astype(int) #(num in classn == 1, 100)
		Rtheta = np.vstack((np.ravel(R, order = 'C'), np.ravel(thetaids, order = 'C'))) #((num in classn == 1)*100, 2)
		unq1, ind1, inv1, cnt1 = np.unique(Rtheta, return_index=True, return_inverse=True, return_counts=True, axis=1)
		counts = np.ndarray.max((cnt1[inv1])[(np.arange(R.size)).reshape(R.shape[0],R.shape[1])], axis = 1)
		
		intensity = 0*inFile.intensity
		intensity[classn == 1] = counts

		out = File("jsd"+file_name, mode = "w", header = inFile.header)
		out.points = inFile.points
		out.intensity = intensity #100*np.absolute((inFile.z- df['Z']))
		#classn[(classn == 1) & (X[inv,3]<0.1) & (inFile.return_num == inFile.num_returns)] = 0
		out.classification = classn
		out.close()
		os.chdir("..")
		print(file_name, "done")


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
