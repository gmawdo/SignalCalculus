import os
from laspy.file import File
import numpy as np
import time
f = open("TIMINGS.txt", "w+")
for big_las_file in os.listdir():
	if big_las_file[-4:] == ".las":
		start = time.time()
		inFile = File(big_las_file, mode = "r")
		new_dir = "TILES_"+big_las_file[:-4]
		os.mkdir(new_dir)
		gps_time = inFile.gps_time
		q_values = np.linspace(0,1,51)[1:] #break [0.0,1.0] into 50 equal segments (i.e. 51 ordinates)
		time_quantiles = np.array(list(np.quantile(gps_time, q = a) for a in q_values))
		digits = np.digitize(gps_time, time_quantiles) # assigns numbers 0-50, with 50 an erroneous value
		digits[digits == 50] = 49
		os.chdir(new_dir)
		for item in range(50):
			if (digits == item).any():
				outFile = File("TILE"+str(item)+"_"+big_las_file, mode = "w", header = inFile.header)
				outFile.points = inFile.points[digits == item]
				outFile.close()
		inFile.close()
		end = time.time()
		print(big_las_file)
		os.chdir("..")
	f.write(big_las_file[-4:]+","+str(end-start)+"\n")
f.close()
