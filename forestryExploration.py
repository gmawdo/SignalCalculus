from laspy.file import File
import numpy as np
inFile = File("OLD-TEST-FILES/Forestry_sample_Ireland_ITM95_co-ords.las", mode = "rw")

x = inFile.x
y = inFile.y
z = inFile.z
rn = inFile.return_num
nr = inFile.num_returns
classification = inFile.classification
classification = 0*classification
working_class = 2
working_class_intensity = 2

peak = classification != classification
classified = classification != 0
unclassified = classification == 0
all_classified = classified.all()
peak_distance = np.full(x.shape, np.inf)
peaks = np.arange(len(inFile))

threshold = 3

outFile = File("forestryExperiment.las", mode = "w", header = inFile.header)
outFile.points = inFile.points

intensity = 0*inFile.intensity

while not(all_classified):
	classified = classification != 0
	all_classified = classified.all()
	unclassified = classification == 0

	coords = np.stack((x,y,z))
	z_unclassified = z[unclassified]
	A_z = np.argmax(z_unclassified)
	z_A = z_unclassified[A_z]

	distance = np.sqrt(np.sum((coords[0:2,:]-coords[0:2,unclassified][0:2,A_z][0:2,None])**2,axis=0))
	mine = distance<np.minimum(threshold, peak_distance)

	clash_condition = (z>z_A) & classified & (distance < threshold)
	ground_condition = (rn == nr) & (distance < threshold)

	if (ground_condition[distance<threshold]).all():
		classification[distance<threshold]=1 #ground being given class 1
		intensity[distance<threshold]=1
	else:
		if (clash_condition).any():
			closest_clash = np.argmin(distance[clash_condition])
			closest_clash_class = classification[clash_condition][closest_clash]
			classification[mine] = closest_clash_class
			working_peak= peaks[clash_condition][closest_clash]
			peaks[mine] = working_peak
			peak_distance[mine] = np.sqrt(np.sum((coords[0:2,mine]-coords[0:2,working_peak][0:2,None])**2,axis=0))
		else:
			classification[mine]=working_class
			intensity[mine] = working_class_intensity
			working_class_intensity = working_class_intensity+1
			working_class = max((working_class+1)%31,2)
			peak_unclassified = peak[unclassified]
			peak_unclassified[A_z]=True
			peak[unclassified]=peak_unclassified
			peak_distance[mine]=distance[mine]
			peaks[mine] = peaks[unclassified][A_z]
		classification[peak] = 31
		outFile.classification = classification%32
		outFile.intensity = intensity


print("Number of peaks:", len(np.unique(classification[classification>1])))
outFile.close()

