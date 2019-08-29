import numpy as np 
from laspy.file import File 
from sklearn.neighbors import NearestNeighbors
import time

start = time.time()

file_name = "Cube.las"
k = 50
C = 10
C_as_intensity = True

inFile = File(file_name)
coords = np.stack((inFile.x, inFile.y, inFile.z), axis = 1) 
nbhrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(coords)
distances, indices = nbhrs.kneighbors(coords)
V = np.mean(coords[indices, :] - coords[:,None,:], axis = 1) #should be same shape as coords
R = np.amax(V, axis = 0) - np.amin(V, axis = 0)
print(np.absolute(R))
print(np.absolute(V))
edge = (np.absolute(R)<C*np.absolute(V)).any(axis = 1)

out = File("edges_"+str(k)+"_"+str(C)+"_"+file_name, mode ="w", header = inFile.header)
out.points = inFile.points
classn = np.zeros(len(inFile), dtype = int)
classn[edge] = 1
out.classification = classn
if C_as_intensity:
	out.intensity = np.amin(np.absolute(R)/np.absolute(V), axis = 1)
out.close()

end = time.time()

print(file_name, str((end-start)/60))
