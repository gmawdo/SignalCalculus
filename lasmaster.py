# lasmaster.py -- A module of all functions used in lidar research, v2

# This module requires np, scikit-learn and laspy to be installed

from laspy.file import File
import numpy as np
import numpy.linalg as LA
from sklearn.neighbors import NearestNeighbors

# G E N E R A T O R   W H I C H   C R E A T E S   M E S H   O F   T W O   L I S T S 
def two_mesh(L,K):
	for i in L:
		for j in K:
			yield [i,j]

# G E N E R A T O R   W H I C H   C R E A T E S   M E S H   O F   T W O   L I S T S 
def three_mesh(L_1,L_2,L_3):
	for i in L_1:
		for j in L_2:
			for k in L_3:
				yield [i,j,k]			

# N E A R E S T   N E I G H B O U R   D I S T A N C E S
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Requires scikit-learn to be installed
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
	out_file.intensity = distances[:,0]
	out_file.close()			

# M A K E   S E N S I B L E   F O R   D I S P L A Z
def makesensible(vector):
	if max(vector) == 0:
		return 0
	else:
		return (1000/(np.quantile(vector,0.9)))*vector

# E I G E N V A L U E S   &   D E T E R M I N A N T   O F   C O V A R I A N C E   M  A T R I X  
# U S I N G   N E A R E S T   N E I G H B O U R S
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is four las files with 1000*eigenvalues and 1000*determinant represented by intensity
# Window is a sphere with chosen radius
def eignn(file_name, num_neighbours = 8):
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array))
	nhbrs = NearestNeighbors(n_neighbors = num_neighbours,algorithm = "kd_tree").fit(np.transpose(coords))
	distances, indices = nhbrs.kneighbors()
	points_for_each_calculation = coords[:,indices].transpose(1,0,2) # creates a 3D tensor of shape (len(in_file), 3, num_neighbours)
	# this tensor contains the n_neighbours nearest neigbours for each point
	eig_storage = np.empty((len(in_file),3))
	for i in range(len(in_file)):
		eig_storage[i,:] = LA.eigvalsh(np.cov(points_for_each_calculation[i]))
	out_file0 = File(file_name+"EigenValue0By"+str(num_neighbours).zfill(3)+"Nbrs.las", mode = "w", header = in_file.header)
	out_file0.points = in_file.points
	out_file1 = File(file_name+"EigenValue1By"+str(num_neighbours).zfill(3)+"Nbrs.las", mode = "w", header = in_file.header)
	out_file1.points = in_file.points
	out_file2 = File(file_name+"EigenValue2By"+str(num_neighbours).zfill(3)+"Nbrs.las", mode = "w", header = in_file.header)
	out_file2.points = in_file.points
	out_file3 =File(file_name+"DeterminantBy"+str(num_neighbours).zfill(3)+"Nbrs.las", mode = "w", header = in_file.header)
	out_file3.points = in_file.points
	out_file0.intensity = makesensible(eig_storage[:,0])
	out_file1.intensity = makesensible(eig_storage[:,1])
	out_file2.intensity = makesensible(eig_storage[:,2])
	out_file3.intensity = makesensible((out_file0.intensity)*(out_file1.intensity)*(out_file2.intensity))
	out_file0.close()
	out_file1.close()
	out_file2.close()
	out_file3.close()

	
				
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
	out_file.intensity = np.array(intensities)
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
	
# L I N E A R   R E G R E S S I O N 
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a sphere with chosen radius
def linreg(file_name, radius):
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
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
			x_for_calc = in_file.x[indices_for_calc]
			y_for_calc = in_file.y[indices_for_calc]
			z_for_calc = in_file.z[indices_for_calc]
			coords_for_calc = np.vstack((x_for_calc,y_for_calc,z_for_calc))
			covmat = np.cov(coords_for_calc)
			if covmat[0,0]*covmat[1,1]*covmat[2,2]==0:
				regression = 1
			else:
				regression = abs(covmat[0,1]*covmat[1,2]*covmat[2,0])/(np.sqrt(covmat[0,0]*covmat[1,1]*covmat[2,2]))
			intensities.append(1000*regression)
		else:
			continue
	N = len(xs)
	out_file = File(file_name+"LinearRegressionRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.x = np.array(xs)
	out_file.y = np.array(ys)
	out_file.z = np.array(zs)
	out_file.return_num = np.array([1]*N)
	out_file.num_returns = np.array([2]*N)
	out_file.raw_classification = np.array([1]*N)
	out_file.intensity = np.array(intensities)
	out_file.close()
	
	
# P L A N A R   R E G R E S S I O N 
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is a las file with function represented by intensity
# Window is a sphere with chosen radius
def coplanreg(file_name, radius):
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
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
			x_for_calc = in_file.x[indices_for_calc]
			y_for_calc = in_file.y[indices_for_calc]
			z_for_calc = in_file.z[indices_for_calc]
			coords_for_calc = np.vstack((x_for_calc,y_for_calc,z_for_calc))
			covmat = np.cov(coords_for_calc)
			exp1 = covmat[0,0]*covmat[1,1]*covmat[2,2]+2*covmat[0,1]*covmat[1,2]*covmat[2,0]
			exp2 = covmat[0,0]*covmat[1,2]*covmat[1,2]+covmat[1,1]*covmat[2,0]*covmat[2,0]+covmat[2,2]*covmat[0,1]*covmat[0,1]
			if exp1 == 0:
				regression = 1
			else:
				regression = exp2/exp1
			intensities.append(1000*regression)
		else:
			continue
	N = len(xs)
	out_file = File(file_name+"CoplanarRegressionRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
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
	
# E I G E N V A L U E S   &   D E T E R M I N A N T   O F   C O V A R I A N C E   M A T R I X 
# File name is the string name of the .las file, e.g. if the file is called "TestArea.las", enter "TestArea"
# Input a las file and a radius
# Output is four las files with 1000*eigenvalues and 1000*determinant represented by intensity
# Window is a sphere with chosen radius
def eigenvalues(file_name, radius):
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
	intensities0 = []
	intensities1 = []
	intensities2 = []
	intensities3 = []
	for dex in lattice:
		x_test=min_x+h*(dex[0]+0.5)
		y_test=min_y+h*(dex[1]+0.5)
		z_test=min_z+h*(dex[2]+0.5)
		indices_for_calc = (in_file.x-x_test)**2 + (in_file.y-y_test)**2 + (in_file.z-z_test)**2 < radius**2
		if sum(indices_for_calc)>3:
			xs.append(x_test)
			ys.append(y_test)
			zs.append(z_test)
			x_for_calc = in_file.x[indices_for_calc]
			y_for_calc = in_file.y[indices_for_calc]
			z_for_calc = in_file.z[indices_for_calc]
			coords_for_calc = np.vstack((x_for_calc,y_for_calc,z_for_calc))
			cov_mat = np.cov(coords_for_calc)
			evals, evecs = LA.eigh(cov_mat)
			intensities0.append(1000*evals[0])
			intensities1.append(1000*evals[1])
			intensities2.append(1000*evals[2])
			intensities3.append(1000*evals[0]*evals[1]*evals[2])
		else:
			continue
	N = len(xs)
	out_file0 = File(file_name+"EigenValue0Radius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.x = np.array(xs)
	out_file0.y = np.array(ys)
	out_file0.z = np.array(zs)
	out_file0.return_num = np.array([1]*N)
	out_file0.num_returns = np.array([2]*N)
	out_file0.raw_classification = np.array([1]*N)
	out_file0.intensity = np.array(intensities0)
	out_file0.close()
	out_file1 = File(file_name+"EigenValue1Radius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file1.x = np.array(xs)
	out_file1.y = np.array(ys)
	out_file1.z = np.array(zs)
	out_file1.return_num = np.array([1]*N)
	out_file1.num_returns = np.array([2]*N)
	out_file1.raw_classification = np.array([1]*N)
	out_file1.intensity = np.array(intensities1)
	out_file1.close()
	out_file2 = File(file_name+"EigenValue2Radius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file2.x = np.array(xs)
	out_file2.y = np.array(ys)
	out_file2.z = np.array(zs)
	out_file2.return_num = np.array([1]*N)
	out_file2.num_returns = np.array([2]*N)
	out_file2.raw_classification = np.array([1]*N)
	out_file2.intensity = np.array(intensities2)
	out_file2.close()
	out_file3 = File(file_name+"DeterminantRadius"+str(int(radius)).zfill(2)+"_"+str(int(100*(radius-int(radius)))).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file3.x = np.array(xs)
	out_file3.y = np.array(ys)
	out_file3.z = np.array(zs)
	out_file3.return_num = np.array([1]*N)
	out_file3.num_returns = np.array([2]*N)
	out_file3.raw_classification = np.array([1]*N)
	out_file3.intensity = np.array(intensities3)
	out_file3.close()