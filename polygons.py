from laspy.file import File 
import numpy as numpy
from sklearn.cluster import DBSCAN

tile = ""
voxel_size = 0.5

inFile = File(tile)
x = inFile.x
y = inFile.y
z = inFile.z
classn = inFile.classification

coords = np.stack((x,y,z), axis = 1)
tree = (classn == 5) | (classn == 4) | (classn == 3)

clustering = DBSCAN(eps=0.5, min_samples=1).fit(coords[tree, :])
labels = clustering.labels_

for label in np.unique(labels):
	condition = np.zeros(len(inFile), dtype = bool)
	condition[tree] = labels == label
	unq, ind, inv, cnt = np.unique(np.floor(coords[condition]/voxel_size).astype(int), return_index=True, return_inverse=True, return_counts=True, axis = 0)
	for item in range(ind.size):
		interior1 = ind[np.all((unq[item] - unq == 1)|(unq[item] - unq == 1), axis = 1)].size == 9
		interior2 = ind[np.all((unq - unq[item] == 2)|(unq - unq[item] == 2), axis = 1)].size == 9
		interior = interior1 & interior2

