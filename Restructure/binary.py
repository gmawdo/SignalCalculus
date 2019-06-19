import os
from laspy.file import File
import numpy as np
from sklearn.neighbors import NearestNeighbors
import lasmaster as lm

for file in os.listdir():
	if file[:4] == "T200" and ("NFL" in file):
		lm.wpd_attr(file)