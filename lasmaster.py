# lasmaster.py -- A module of all functions used in lidar research, v2

# This module requires np, scipy, scikit-learn and laspy to be installed

from laspy.file import File
import numpy as np
import numpy.linalg as LA
from sklearn.neighbors import NearestNeighbors

# M A K E   A   V E C T O R   S E N S I B L E   F O R   D I S P L A Z   P L O T T I N G
# This function takes a vector (which should consist of non-negative values) and outputs a vector whose 90th percentile value is 1000
# Unless the vector is zero, in which case the only sensible thing to do is output zero
# The reason we do this is to get a reasonable plot in Displaz and so we can use a fixed color scale
# Thus we can ignore outliers which occur in the upper percentiles
def makesensible(vector):
	if max(vector) == 0:
		return 0
	else:
		return (1000/(np.quantile(vector,0.9)))*vector
				
# N E A R E S T   N E I G H B O U R   D I S T A N C E S
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input just a LAS file
# Output is a LAS file with signal value represented by intensity
def nnbd(file_name):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.X
	y_array = in_file.Y
	z_array = in_file.Z
	coords = np.transpose(np.vstack((x_array,y_array,z_array)))
	nhbrs = NearestNeighbors(n_neighbors = 2,algorithm = "kd_tree").fit(coords)
	distances, indices = nhbrs.kneighbors()
	out_file = File(file_name+"NNDistancesRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points
	out_file.intensity = makesensible(distances[:,0])
	out_file.close()			
	
# L I N E A R   R E G R E S S I O N
# U S I N G   N E A R E S T   N E I G H B O U R S
# Only considers neighbours inside a chosen radius
# Deletes bad quality points which don't have enough neighbours
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is four las files with 1000*eigenvalues and 1000*determinant represented by intensity
# Window is a sphere with chosen radius
def statz(file_name, min_num_neighbours = 4, num_neighbours = 8, radius = 1):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array))
	nhbrs = NearestNeighbors(n_neighbors = num_neighbours, algorithm = "kd_tree").fit(np.transpose(coords))
	print("Training Done")
	distances, indices = nhbrs.kneighbors(np.transpose(coords))
	print("Distances and nearest neighbours found")
	points_for_each_calculation = coords[:,indices].transpose(1,0,2)
	linreg_storage = np.empty((len(in_file)))
	planreg_storage = np.empty((len(in_file)))
	eig_storage = np.empty((len(in_file),3))
	coicou_storage = np.empty((len(in_file)))
	for i in range(len(in_file)):
		if sum(distances[i]<radius) >= min_num_neighbours:
			covmat = np.cov(points_for_each_calculation[i][:,distances[i]<radius], bias = True)
			exp1 = covmat[0,0]*covmat[1,1]*covmat[2,2]+2*covmat[0,1]*covmat[1,2]*covmat[2,0]
			exp2 = covmat[0,0]*covmat[1,2]*covmat[1,2]+covmat[1,1]*covmat[2,0]*covmat[2,0]+covmat[2,2]*covmat[0,1]*covmat[0,1]
			if (covmat[0,0]*covmat[1,1]*covmat[2,2])==0:
				linreg_storage[i] = 1
			else:
				linreg_storage[i] = abs(covmat[0,1]*covmat[1,2]*covmat[2,0])/(covmat[0,0]*covmat[1,1]*covmat[2,2])
			if exp1 == 0:
				planreg_storage[i] = 1
			else:
				planreg_storage[i] = exp2/exp1
			eig_storage[i,:] = LA.eigvalsh(np.cov(points_for_each_calculation[i][:,distances[i]<radius]))
		else:
			linreg_storage[i]=0
			planreg_storage[i]=0
			eig_storage[i,:] = np.empty((1,3))
		CoU = np.mean(points_for_each_calculation[i][:,distances[i]<radius], axis=1)
		CoI = np.average(points_for_each_calculation[i][:,distances[i]<radius], axis = 1, weights = in_file.intensity[indices])
		coicou_storage[i]=np.sqrt(np.sum((CoU-CoI)**2))
	print("Core calculations complete ... preparing LAS files")
	KEEP = np.sum(distances<radius,axis=1)>=min_num_neighbours
	out_file = File(file_name+"LinRegMax"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points[KEEP]
	out_file.intensity = 1000*linreg_storage[KEEP]
	out_file.close()
	out_file0 = File(file_name+"Eig0BMax"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.points = in_file.points[KEEP]
	out_file1 = File(file_name+"Eig1Max"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file1.points = in_file.points[KEEP]
	out_file2 = File(file_name+"Eig2Max"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file2.points = in_file.points[KEEP]
	out_file3 = File(file_name+"DetMax"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file3.points = in_file.points[KEEP]
	out_file4 =File(file_name+"Eig10Max"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file4.points = in_file.points[KEEP]
	out_file5 = File(file_name+"Eig21Max"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file5.points = in_file.points[KEEP]
	e0 = eig_storage[:,0][KEEP]
	e1 = eig_storage[:,1][KEEP]
	e2 = eig_storage[:,2][KEEP]
	d = e0*e1*e2
	e10 = e1-e0
	e21 = e2-e1
	out_file0.intensity = makesensible(e0)
	out_file1.intensity = makesensible(e1)
	out_file2.intensity = makesensible(e2)
	out_file3.intensity = makesensible(d)
	out_file4.intensity = makesensible(e10)
	out_file5.intensity = makesensible(e21)
	out_file0.close()
	out_file1.close()
	out_file2.close()
	out_file3.close()
	out_file4.close()
	out_file5.close()
	out_file6 = File(file_name+"PlanRegMax"+str(num_neighbours).zfill(3)+"Min"+str(min_num_neighbours).zfill(3)+"Rad"+str(int(radius)).zfill(2)+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file6.points = in_file.points[KEEP]
	out_file6.intensity = 1000*planreg_storage[KEEP]
	out_file6.close()

	
# H O U G H   T R A N S F O R M
def Hough(file_name):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	a = np.linspace(0, np.pi, num = 1000, endpoint = False)
	r = np.outer(np.cos(a),x_array)+np.outer(np.sin(a),y_array)
	distances = np.sqrt(in_file.x**2+in_file.y**2)
	min_r = min(distances)
	max_r = max(distances)
	out_file = File(file_name+"HoughWinner.las", mode = "w", header = in_file.header)
	print("Calculations round 1 done")
	if np.cos(a_winner) == 0:
		x_s = np.array([min(x_array)+(max(x_array)-min(x_array))*i/1000 for i in range(1000)])
		y_s = np.array([r_winner]*1000)
	else:
		y_s = np.array([min(y_array)+(max(y_array)-min(y_array))*i/1000 for i in range(1000)])
		x_s = np.array([(r_winner-np.sin(a_winner)*(min(y_array)+(max(y_array)-min(y_array))*i/1000))/np.cos(a_winner) for i in range(1000)])
	print("Calculations round 2 done ... making output")
	print("a_winner = "+str(a_winner))
	print("r_winner = "+str(r_winner))
	out_file.x = x_s
	out_file.y = y_s
	out_file.z = np.array([np.mean(in_file.z)]*1000)
	out_file.return_num = np.array([1]*1000)
	out_file.num_returns = np.array([1]*1000)
	out_file.raw_classification = np.array([6]*1000)
	out_file.intensity = np.array([1000]*1000)
	out_file.close()
				
# P L A N   V I E W   L O C A L   P O I N T   C O U N T
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a cylinder with chosen radius
def ptct(file_name, radius):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	max_x = max(x_array)
	max_y = max(y_array)
	min_x = min(x_array)
	min_y = min(y_array)
	h = np.sqrt(2)*radius
	N_x = int(max(np.ceil((max_x-min_x)/h),1))
	N_y = int(max(np.ceil((max_y-min_y)/h),1))
	lattice = two_mesh(range(N_x),range(N_y))
	xs = []
	ys = []
	zs = []
	intensities = []
	for dex in lattice:
		x_0 = min_x+h*(dex[0]+0.5)
		y_0 = min_y+h*(dex[1]+0.5)
		indices_for_calc = (in_file.x-x_0)**2 + (in_file.y-y_0)**2 < radius**2
		if True in indices_for_calc:
			z_0 = sum(indices_for_calc)/(radius**2)
			xs.append(x_0)
			ys.append(y_0)
			zs.append(z_0)
		else:
			continue
	out_file = File(file_name+"PtCtRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.x = np.array(xs)
	out_file.y = np.array(ys)
	out_file.intensity = np.array(zs)
	out_file.z = np.array([np.mean(in_file.z)]*N)
	out_file.return_num = np.array([1]*N)
	out_file.num_returns = np.array([2]*N)
	out_file.raw_classification = np.array([1]*N)
	out_file.close()

# L O C A L   M I N - M A X   O F   I N T E N S I T Y 
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a sphere with chosen radius
def rangeint(file_name, radius):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	h = np.sqrt(3)*radius/2
	N_x = int(max(np.ceil((max_x-min_x)/h),1))
	N_y = int(max(np.ceil((max_y-min_y)/h),1))
	N_z = int(max(np.ceil((max_z-min_z)/h),1))
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	intensities = []
	for dex in lattice:
		x_test=min_x+h*(dex[0]+0.5)
		y_test=min_y+h*(dex[1]+0.5)
		z_test=min_z+h*(dex[2]+0.5)
		indices_for_calc = (in_file.x-x_test)**2 + (in_file.y-y_test)**2 + (in_file.z-z_test)**2 < radius**2
		if True in indices_for_calc:
			intensities.append(max(in_file.intensity[indices_for_calc])-min(in_file.intensity[indices_for_calc]))
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
		else:
			continue
	N = len(xs)
	out_file = File(file_name+"IntensityRangeRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.x = np.array(xs)
	out_file.y = np.array(ys)
	out_file.z = np.array(zs)
	out_file.return_num = np.array([1]*N)
	out_file.num_returns = np.array([2]*N)
	out_file.raw_classification = np.array([1]*N)
	out_file.intensity = np.array(makesensible(intensities))
	out_file.close()

# L O C A L   S T A N D A R D   D E V I A T I O N   O F   I N T E N S I T Y
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a sphere with chosen radius
def stdint(file_name, radius):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	h = np.sqrt(3)*radius/2
	N_x = int(max(np.ceil((max_x-min_x)/h),1))
	N_y = int(max(np.ceil((max_y-min_y)/h),1))
	N_z = int(max(np.ceil((max_z-min_z)/h),1))
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	intensities = []
	for dex in lattice:
		x_test=min_x+h*(dex[0]+0.5)
		y_test=min_y+h*(dex[1]+0.5)
		z_test=min_z+h*(dex[2]+0.5)
		indices_for_calc = (in_file.x-x_test)**2 + (in_file.y-y_test)**2 + (in_file.z-z_test)**2 < radius**2
		if sum(indices_for_calc)>1:
			intensities.append(np.std(in_file.intensity[indices_for_calc]))
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
		else:
			continue
	N = len(xs)
	out_file = File(file_name+"StDevIntensityRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.x = np.array(xs)
	out_file.y = np.array(ys)
	out_file.z = np.array(zs)
	out_file.return_num = np.array([1]*N)
	out_file.num_returns = np.array([2]*N)
	out_file.raw_classification = np.array([1]*N)
	out_file.intensity = np.array(intensities)
	out_file.close()
	
# C O I   V S   C O U 
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a sphere with chosen radius
def coicou(file_name, radius):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	h = np.sqrt(3)*radius/2
	N_x = int(max(np.ceil((max_x-min_x)/h),1))
	N_y = int(max(np.ceil((max_y-min_y)/h),1))
	N_z = int(max(np.ceil((max_z-min_z)/h),1))
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	intensities = []
	for dex in lattice:
		x_test=min_x+h*(dex[0]+0.5)
		y_test=min_y+h*(dex[1]+0.5)
		z_test=min_z+h*(dex[2]+0.5)
		indices_for_calc = (in_file.x-x_test)**2 + (in_file.y-y_test)**2 + (in_file.z-z_test)**2 < radius**2
		if True in indices_for_calc:
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
			intsformean = in_file.intensity[indices_for_calc]
			xsformean = in_file.x[indices_for_calc]
			ysformean = in_file.y[indices_for_calc]
			zsformean = in_file.z[indices_for_calc]
			Is = intsformean.astype(float)
			CoU = np.array([np.mean(xsformean),np.mean(xsformean),np.mean(xsformean)])
			CoI = np.array([np.mean(xsformean*Is)/np.mean(Is),np.mean(xsformean*Is)/np.mean(Is),np.mean(xsformean*Is)/np.mean(Is)])
			regression=np.sqrt(np.sum((CoU-CoI)**2))/radius
			intensities.append(1000*regression)
		else:
			continue
	N = len(xs)
	out_file = File(file_name+"CoICoURadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.x = np.array(xs)
	out_file.y = np.array(ys)
	out_file.z = np.array(zs)
	out_file.return_num = np.array([1]*N)
	out_file.num_returns = np.array([2]*N)
	out_file.raw_classification = np.array([1]*N)
	out_file.intensity = np.array(intensities)
	out_file.close()
