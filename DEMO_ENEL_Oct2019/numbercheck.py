import numpy as np
from laspy.file import File
import os
os.chdir("TILES")
for tile in os.listdir():
	if tile[:3] == "DH5":
			inFile = File(tile)
			n = len(inFile)
			os.chdir("RESULTS")
			for tile_result in os.listdir():
					if tile_result == tile:
							inFile_result = File(tile_result)
							n_result = len(inFile_result)
							print(tile)
							print("I size", n)
							print("O size", n_result)
							print(n==n_result)
							print(" ")
			os.chdir("..")