# S T A T I S T I C A L   S I G N A L 

import numpy as np
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
import numpy.linalg as LA
import time

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
def nnd(file_name):
	start = time.time()
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
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")


def less(file_name):
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	out_file = File(file_name+"Less.las", mode = "w", header = in_file.header)
	out_file.points = in_file.points[in_file.return_num<in_file.num_returns]
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def stats(file_name, m = 4, k= 50, radius = 0.75, clip = 0.99):
# in comments below N is num pts in file
# parenthetic comments indicate shape of array
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array)) # (3,N)
	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)) 
	print("Training done...")
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (N,k)
	print("Distances and indices found...")
	neighbours = coords[:,indices] # (3,N,k)
	keeping = distances<radius # (N,k)
	Ns = np.sum(keeping, axis = 1) # (N)
	means = np.sum(neighbours*keeping/Ns[None,:,None], axis = 2) # (3,N)
	raw_deviations = (neighbours - means[:,:,None])*keeping # (3,N,k)
	end = time.time()
	print("Time taken so far: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
	print("Means taken...")
	xy_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	yz_covs = np.sum(raw_deviations[1,:,:]*raw_deviations[2,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	zx_covs = np.sum(raw_deviations[2,:,:]*raw_deviations[0,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	xx_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[0,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	yy_covs = np.sum(raw_deviations[1,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	zz_covs = np.sum(raw_deviations[2,:,:]*raw_deviations[2,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	evals = LA.eigvalsh(np.swapaxes(np.dstack((np.vstack((xx_covs,xy_covs,zx_covs)),np.vstack((xy_covs,yy_covs,yz_covs)),np.vstack((zx_covs,yz_covs,zz_covs)))),0,1))
	good_pts = Ns>=m
	pt_ct = np.sum(keeping, axis = 1)
	exp1 = xx_covs[good_pts]*yy_covs[good_pts]*zz_covs[good_pts]+2*xy_covs[good_pts]*yz_covs[good_pts]*zx_covs[good_pts]
	exp2 = xx_covs[good_pts]*yz_covs[good_pts]*yz_covs[good_pts]+yy_covs[good_pts]*zx_covs[good_pts]*zx_covs[good_pts]+zz_covs[good_pts]*xy_covs[good_pts]*xy_covs[good_pts]
	lin_regs = abs(xy_covs[good_pts]*yz_covs[good_pts]*zx_covs[good_pts]/(xx_covs[good_pts]*yy_covs[good_pts]*zz_covs[good_pts]))
	xy_lin_regs = abs(xy_covs[good_pts]/np.sqrt(xx_covs[good_pts]*yy_covs[good_pts]))
	plan_regs = exp2/exp1
	lin_regs[np.logical_or(np.isnan(lin_regs),np.isinf(lin_regs))]=1
	xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs),np.isinf(xy_lin_regs))]=1
	plan_regs[np.logical_or(np.isnan(plan_regs),np.isinf(plan_regs))]=1
	end = time.time()
	print("Time taken so far: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
	print("Calculations done... making LAS files")
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	C_int = int(clip)
	C_rat = int(100*(clip-int(clip)))
	points = in_file.points[good_pts]
	out_file0 = File(file_name+"LinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.points = points
	out_file0.intensity = 1000*lin_regs
	out_file0.close()
	end = time.time()
	out_file1 = File(file_name+"XYLinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file1.points = points
	out_file1.intensity = 1000*xy_lin_regs
	out_file1.close()
	out_file2 = File(file_name+"PlanRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file2.points = points
	out_file2.intensity = 1000*plan_regs
	out_file2.close()
	out_file3 = File(file_name+"PtCtMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file3.points = points
	out_file3.intensity = 1000*((pt_ct/k)[good_pts])
	out_file3.close()	
	end = time.time()
	out_file3 = File(file_name+"RuggednessMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file3.points = points
	out_file3.intensity = makesensible(np.sqrt(zz_covs[good_pts]))
	out_file3.close()
	end = time.time()
	out_file4 = File(file_name+"DiscXYLinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+"Clip"+str(C_int)+"_"+str(C_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file4.points = in_file.points[good_pts][xy_lin_regs>clip]
	out_file4.close()
	out_file5 = File(file_name+"FewPtsMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file5.points = in_file.points[np.logical_and(pt_ct<k,good_pts)]
	out_file5.close()
	out_file7 = File(file_name+"Eigenvalue0Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file7.points = in_file.points[good_pts]
	out_file7.intensity = makesensible(evals[:,0][good_pts])
	out_file7.close()
	out_file8 = File(file_name+"Eigenvalue1Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file8.points = in_file.points[good_pts]
	out_file8.intensity = makesensible(evals[:,1][good_pts])
	out_file8.close()
	out_file9 = File(file_name+"Eigenvalue2Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file9.points = in_file.points[good_pts]
	out_file9.intensity = makesensible(evals[:,2][good_pts])
	out_file9.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def deforest(file_name,clip=1):
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array)) # (2,N)
	less = in_file.points[in_file.return_num<in_file.num_returns]
	coordsless = coords[:,in_file.return_num<in_file.num_returns]
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(coords[:,np.logical_and(in_file.return_num==4,in_file.num_returns==5)]))
	print("Training done...")
	distances, indices = nhbrs.kneighbors(np.transpose(coordsless)) # (N,k)
	print("Distances found...")
	C_int = int(clip)
	C_rat = int(100*(clip-int(clip)))
	out_file = File(file_name+"Deforest"+"Clip"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = less[distances[:,0]>clip]
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def xyptdens(file_name, radius = 1):
	from sklearn.neighbors import KDTree
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	coords = np.vstack((x_array,y_array)) # (2,N)
	tree = KDTree(np.transpose(coords), leaf_size=2)
	pt_ct = tree.query_radius(np.transpose(coords), r=radius, count_only=True)
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	out_file = File(file_name+"XYPtDensityRadius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points
	out_file.intensity = makesensible(pt_ct/radius**2)
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def ptdens(file_name, radius = 1):
	from sklearn.neighbors import KDTree
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array, z_array)) # (2,N)
	tree = KDTree(np.transpose(coords), leaf_size=2)
	pt_ct = tree.query_radius(np.transpose(coords), r=radius, count_only=True)
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	out_file = File(file_name+"PtDensityRadius"+str(R_int).zfill(2)+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points
	out_file.intensity = makesensible(pt_ct/radius**3)
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

# N E A R   F L I G H T   L I N E
def nfl(file_name, clip = 100):
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	class10 = inFile.classification==10
	x_array = inFile.x
	y_array = inFile.y
	coords = np.vstack((x_array,y_array))
	x_flight = x_array[class10]
	y_flight = y_array[class10]
	coords_flight = np.vstack((x_flight,y_flight))
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(coords_flight))
	distances, indices = nhbrs.kneighbors(np.transpose(coords))
	out_file = File(file_name+"NFLClip"+str(int(clip)).zfill(3)+"_"+str(int(100*(clip-int(clip)))).zfill(2)+".las", mode = "w", header = in_file.header)
	outfile.points = inFile.points[np.logical_not(distances[:,0]<clip)]
	outfile.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

# W E I G H T S   F R O M   K - N E I G H B O U R S   G R A P H
def nng(file_name, k=1):
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array)) # (3,N)
	graph = kneighbors_graph(np.transpose(coords), n_neighbors = k)
	num_components, labels = connected_components(graph, directed=False, return_labels=True)
	out_file = File(file_name+"Connectivity"+str(k).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points
	out_file.intensity = makesensible(np.bincount(labels)[labels])
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")



