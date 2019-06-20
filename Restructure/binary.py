import os
from laspy.file import File
import numpy as np
from sklearn.neighbors import NearestNeighbors
import lasmaster as lm

for file in os.listdir():
	if file[:4] == "attr" and ("NFL" in file):
		inFile = File(file, mode = "rw")
		inFile.intensity = 1000*inFile.entent
		inFile.close()
