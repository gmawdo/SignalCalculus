import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

theta = np.linspace(0, 2*np.pi, num = 1000)
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

import os
from laspy.file import File

for tile in os.listdir():
	if "classified" in tile and "bb" not in tile and ".las" in tile:
		print(tile)
		inFile = File(tile)
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

		out.classification = classn
		np.savetxt("buildings_"+tile[-4:]+".csv", bldgs, delimiter=",", header = "ID, Area, X_min, X_max, Y_min, Y_max")
		out.close()


