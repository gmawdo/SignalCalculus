import os
import lasmaster as lm
os.chdir("testingLAS")
from laspy.file import File
import numpy as np
for file_name in os.listdir():	
	if file_name[:3]=="hag":
		daFile = File(file_name)
		inFile = File("redatum"+file_name, mode = "w", header = daFile.header)
		inFile.points = daFile.points
		inFile.intensity = 100*np.absolute(daFile.hag)
		inFile.z = inFile.hag
		inFile.close()
