from laspy.file import File
import numpy as np
from sklearn.neighbors import NearestNeighbors

inFile = File("OLD-TEST-FILES/Forestry_sample_Ireland_ITM95_co-ords.las", mode = "rw")


rn = inFile.return_num
nr = inFile.num_returns

not_last_return = (rn != nr)

x = inFile.x[not_last_return]
y = inFile.y[not_last_return]
z = inFile.z[not_last_return]

classification = np.arange(len(inFile))[not_last_return]

outFile = File("forestryExperiment.las", mode = "w", header = inFile.header)
outFile.points = inFile.points

classification[:] = 0

peak = classification != classification
classified = classification != 0
unclassified = classification == 0
all_classified = classified.all()

threshold = 3

while not(all_classified):
	classified = classification != 0
	all_classified = classified.all()
	unclassified = classification == 0

	coords = np.stack((x,y,z))
	z_unclassified = z[unclassified]
	A_z = np.argmax(z_unclassified)
	z_A = z_unclassified[A_z]

	distance = np.sqrt(np.sum((coords[0:2,:]-coords[0:2,unclassified][0:2,A_z][0:2,None])**2,axis=0))
	mine = distance<threshold
	
	z_test = (z>z_A)
	clash_condition = z_test & mine
	clashes = clash_condition.any()
	mine_class = classification[mine]
	classification[mine & np.logical_not(z_test)] = 1
	class_unclassified = classification[unclassified]
	class_unclassified[A_z] = 2-int(clashes)
	classification[unclassified] = class_unclassified
	classified = classification != 0
	all_classified = classified.all()
	unclassified = classification == 0




peak = classification == 2
classification[:] = 0
Classification = np.arange(len(inFile))
classification[peak] = (np.arange(sum(peak))+1).astype(classification.dtype)
Classification[not_last_return] = classification
Peak = Classification != Classification
Peak[not_last_return] = peak

X = inFile.x
Y = inFile.y
Z = inFile.z
c2d = np.vstack((X,Y))

nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c2d[:, Peak]))		
distances2d, indices2d = nhbrs2d.kneighbors(np.transpose(c2d))
Classification = Classification[Peak][indices2d[:,0]]
nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c2d[:, not_last_return]))		
distances2d, indices2d = nhbrs2d.kneighbors(np.transpose(c2d))
Classification[distances2d[:,0]>0.25]=0

max_class = max(Classification)
red = np.random.random(max_class+1)
blue = np.random.random(max_class+1)
green = np.random.random(max_class+1)
red[0] = 0
blue[0] = 0
green[0] = 0

outFile.Red = red[Classification]
outFile.Blue = blue[Classification]
outFile.Green = green[Classification]


outFile.close()




'''

from laspy.file import File
import numpy as np
inFile = File("OLD-TEST-FILES/Forestry_sample_Ireland_ITM95_co-ords.las", mode = "rw")

rn = inFile.return_num
nr = inFile.num_returns

not_last_return = (rn != nr)

x = inFile.x[not_last_return]
y = inFile.y[not_last_return]
z = inFile.z[not_last_return]

classification = inFile.classification[not_last_return]
intensity = inFile.intensity[not_last_return]

outFile = File("forestryExperiment.las", mode = "w", header = inFile.header)
outFile.points = inFile.points[not_last_return]

classification = 0*classification
intensity = 0*intensity
working_class = 2
working_class_intensity = 2



peak = classification != classification
classified = classification != 0
unclassified = classification == 0
all_classified = classified.all()
peak_distance = np.full(x.shape, np.inf)
peaks = np.arange(len(x))

threshold = 3




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

	clash_condition = (z>z_A) & classified & mine


	if (clash_condition).any():
		closest_clash = np.argmin(distance[clash_condition])
		closest_clash_class = classification[clash_condition][closest_clash]
		classification[mine] = closest_clash_class
		closest_clash_intensity = intensity[clash_condition][closest_clash]
		intensity[mine] = closest_clash_intensity
		working_peak= peaks[clash_condition][closest_clash]
		peaks[mine] = working_peak
		peak_distance[mine] = np.sqrt(np.sum((coords[0:2,mine]-coords[0:2,working_peak][0:2,None])**2,axis=0))
		classification[A_z] = 1
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
'''

