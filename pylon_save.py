import numpy as np
from laspy.file import File
from sklearn.neighbors import BallTree

tile_condition = lambda x: "attr" in x and "classified" not in x

for tile_name in os.listdir():
	if tile_condition(tile_name):
		start = time.time()
		inFile = File(tile_name,mode = "r")
		Coords = np.stack((inFile.x,inFile.y,inFile.z))
		Std = 