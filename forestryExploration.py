from laspy.file import File
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

start = time.time() 

inFile = File("OLD-TEST-FILES/Forestry_sample_Ireland_ITM95_co-ords.las", mode = "rw")
threshold = 3

rn = inFile.return_num
nr = inFile.num_returns
not_last_return = rn != nr

X = inFile.x
Y = inFile.y
Z = inFile.z
x = X[not_last_return]
y = Y[not_last_return]
z = Z[not_last_return]

Classification = np.zeros(len(inFile), dtype = int)
classification = Classification[not_last_return]

classified = classification != 0
unclassified = classification == 0
all_classified = classified.all()

while not(all_classified):

	coords = np.stack((x,y,z))
	z_unclassified = z[unclassified]
	A_z = np.argmax(z_unclassified)
	z_A = z_unclassified[A_z]

	distance = np.sqrt(np.sum((coords[0:2,:]-coords[0:2,unclassified][0:2,A_z][0:2,None])**2,axis=0))
	mine = distance < threshold
	
	z_test = z > z_A
	clash_condition = z_test & mine
	clashes = clash_condition.any()
	classification[mine & np.logical_not(z_test)] = 1
	class_unclassified = classification[unclassified]
	class_unclassified[A_z] = 2-int(clashes)
	classification[unclassified] = class_unclassified

	classified = classification != 0
	unclassified = classification == 0
	all_classified = classified.all()

peak = classification == 2
classification[:] = 0

number_of_trees = sum(peak)
classification_peak = classification[peak]
classification_peak = 1+np.arange(number_of_trees)
classification[peak] = classification_peak

c2d = np.vstack((x,y))

nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c2d[:, peak]))		
distances2d, indices2d = nhbrs2d.kneighbors(np.transpose(c2d))
classification = ((classification[classified])[indices2d[:,0]])

C2D = np.vstack((X,Y))
Nhbrs2d = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(c2d))
Distances2d, Indices2d = Nhbrs2d.kneighbors(np.transpose(C2D))

Classification = classification[Indices2d[:,0]]
Classification[Distances2d[:,0]>0.5] = 0

ID = classification[peak]
height = x[peak]
radius = x[peak]
CrownX = x[peak]
CrownY = y[peak]
CrownZ = z[peak]
for item in ID:
	height[ID == item] = max(Z[Classification == item])-min(Z[Classification == item])
	radius[ID == item] = max(distances2d[Indices2d[Classification == item, 0],0])

condition = (height > 2.5)
print(sum(condition))
print(len(ID))
print(height[condition])
trees = np.stack((ID, height, radius, CrownX, CrownY, CrownZ), axis = 1)

ID[np.logical_not(condition)] = 0
classification[peak] = ID
classification = ((classification[peak])[indices2d[:,0]])
Classification = classification[Indices2d[:,0]]
Classification[Distances2d[:,0]>0.5] = 0

np.savetxt("trees.csv", trees[:, condition], delimiter=",", header = "ID, Height, Radius, CrownX, CrownY, CrownZ")
outFile = File("forestryExperiment.las", mode = "w", header = inFile.header)
outFile.points = inFile.points

outFile.classification = Classification%32
outFile.close()

end = time.time()
print("Done. Time taken = "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds!")


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

