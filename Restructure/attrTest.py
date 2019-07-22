import os
import lasmaster as lm
os.chdir("testingLAS")
from laspy.file import File
import numpy as np
for file_name in os.listdir():	
	if "Completa" in file_name and not("attr" in file_name):
		lm.example_attr(file_name)
