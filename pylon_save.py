import numpy as np
from laspy.file import File
from sklearn.neighbors import BallTree

tile_condition = lambda x: "attr" in x and "classified" not in x

for tile_name in os.listdir():
	if tile_condition(tile_name):
		start = time.time()
		inFile = File(tile_name, mode = "r")
		Coords = np.stack((inFile.x,inFile.y,inFile.z), axis = 1) # n x 3
		nhbrs = NearestNeighbors(n_neighbors = 50, algorithm = "kd_tree").fit(np.transpose(Coords))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords)) # n x 50
		noise = distances[:, -1] < 0.5
		sigma_XYZ = np.std(Coords[indices[~ noise, :], :], axis = 1) # n x 3
		condition = sigma_XYZ[:, 0] + sigma_XYZ[:, 1] < sigma_XYZ[:, 2] # n
		classn = inFile.classification
		classn[:]  = 0
		classn_sig = classn[ ~ noise]
		classn_sig[condition] = 1
		classn_sig[ ~ condition] = 2
		classn[ ~ noise] = classn_sig
		out = File("tall_thin_" + tile_name)
		out.points = inFile.points
		out.classification = classn
		print(tile_name)
		out.close()






		