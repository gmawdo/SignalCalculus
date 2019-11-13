import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def bbox(tile, N = 1000)

	theta = np.linspace(0, 2*np.pi, num = N)
	S = np.sin(theta)
	C = np.cos(theta)
	R = np.zeros((theta.size, 2, 2))
	R[:, 0, 0] = np.cos(theta)
	R[:, 0, 1] = -np.sin(theta)
	R[:, 1, 0] = np.sin(theta)
	R[:, 1, 1] = np.cos(theta)

	def bb(x, y, z, predicate):
		Coords = np.stack((x,y), axis = 0)
		coords_R = np.matmul(R, Coords[:, predicate])
		x_max = np.amax(coords_R[:, 0, :], axis = -1)
		y_max = np.amax(coords_R[:, 1, :], axis = -1)
		x_min = np.amin(coords_R[:, 0, :], axis = -1)
		y_min = np.amin(coords_R[:, 1, :], axis = -1)
		A = (x_max-x_min)*(y_max-y_min)
		k = np.argmin(A)
		R_min = R[k, : , :]
		Coords_R = np.matmul(R[k, :, :], Coords)
		predicate_bb = (x_min[k]-0.25<=Coords_R[0]) & (y_min[k]-0.25<=Coords_R[1]) & (x_max[k]>=Coords_R[0]-0.25) & (y_max[k]>=Coords_R[1]-0.25) & (max(z[predicate])+0.5>=z) & (min(z[predicate])-0.5<=z)
		return predicate_bb, A[k], min(x[predicate_bb]), max(x[predicate_bb]), min(y[predicate_bb]), max(y[predicate_bb])
		
	x = inFile.x
	y = inFile.y
	z = inFile.z
	coords = np.stack((x,y), axis = 1)
	out = File("bb_"+tile, mode = "w", header = inFile.header)
	out.points = inFile.points
	classn = np.ones(len(inFile), dtype = int)
	classn[:] = inFile.classification[:]
	classn_2_save = classn == 2
	if (classn==2).any():
		clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x,y,z), axis = 1)[classn_2_save, :])
		labels = clustering.labels_
		L = np.unique(labels)
		bldgs = np.empty((L.size, 6))
		i=0
		for item in L:
			predicate = np.zeros(len(inFile), dtype = bool)
			predicate[classn_2_save] = labels == item
			predicate_bb, area, x_min, x_max, y_min, y_max = bb(x, y, z, predicate)
			classn[predicate_bb] = 2
			bldgs[i] = [i, area, x_min, x_max, y_min, y_max]
			i+=1
			np.savetxt("buildings_"+tile[-4:]+".csv", bldgs, delimiter=",", header = "ID, Area, X_min, X_max, Y_min, Y_max")

	if (classn == 0).any() and (classn == 1).any():
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.stack((x,y), axis = 1)[classn == 1,:])
		distances, indices = nhbrs.kneighbors(np.stack((x,y), axis = 1)[classn == 0,:])
		classn0 = classn[classn == 0]
	
		try:
			angle = inFile.ang3
		except:
			angle = inFile.ang2


		
		classn0[(distances[:,0]< 1) & (angle[classn == 0] < 0.2)] = 5
		classn0[(distances[:,0]< 1) & (angle[classn == 0] < 0.2)] = 5
		classn[classn == 0] = classn0

		if (classn == 5).any():
			classn_5_save = classn == 5
			classn5 = classn[classn_5_save]
			unq, ind, inv = np.unique(np.floor(np.stack((x,y), axis = 1)[classn_5_save,:]).astype(int), return_index=True, return_inverse=True, return_counts=False, axis = 0)
			for item in np.unique(inv):
				z_max = np.max(z[classn_5_save][inv == item])
				z_min = np.min(z[classn_5_save][inv == item])
				if (z_max-z_min<5):
					classn5[inv==item]=0
			classn[classn_5_save]=classn5

		
		if (classn==5).any():
					classn_5_save = classn == 5
					clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x,y), axis = 1)[classn_5_save, :])
					labels = clustering.labels_
					L = np.unique(labels)
					
					for item in L:
						predicate = np.zeros(len(inFile), dtype = bool)[(classn==0)|classn_5_save]
						predicate[classn_5_save[(classn==0)|classn_5_save]] = labels == item
						predicate_bb, area, x_min, x_max, y_min, y_max = bb(x[(classn==0)|classn_5_save], y[(classn==0)|classn_5_save], z[(classn==0)|classn_5_save], predicate)
						classn05 = classn[(classn==0)|classn_5_save]
						classn05[predicate_bb] = 5
						classn[(classn==0)|classn_5_save]=classn05

	out.classification = classn
	out.close()


