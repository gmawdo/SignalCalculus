import os
from laspy.file import File
for big_las_file in os.listdir():
	if big_las_file[-4:] == ".las":
		inFile = File(big_las_file, mode = "r")
		new_dir = "TILES_"+big_las_file[:-4]
		os.mkdir(new_dir)
		gps_time = inFile.gps_time
		q_values = np.linspace(0,1,51)[1:] #break [0.0,1.0] into 50 equal segments (i.e. 51 ordinates)
		digits = np.digitize(gps_tile, q_values) # assigns numbers 0-50, with 50 an erroneous value
		digits[digits == 50] = 49
		os.chdir(new_dir)
		for item in range(50):
			if (digits == item).any():
				outFile = File("TILE"+str(item)+"_"+big_las_file, mode = "w", header = inFile.header)
				outFile.points = inFile.points[digits == item]
				outFile.close()
		os.chdir("..")
