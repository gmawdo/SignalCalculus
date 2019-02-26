# S T A T I S T I C A L   S I G N A L 

import numpy as np
import numpy.linalg as LA
import time
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KDTree
from scipy.stats import mode
from scipy.sparse.csgraph import connected_components


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

def stats(file_name, m = 4, k= 50, radius = 0.75, clip = 0.99, thresh = 0.001):
# in comments below N is num pts in file
# parenthetic comments indicate shape of array
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	t_array = in_file.gps_time
	coords = np.vstack((x_array,y_array,z_array)) # (3,N)
	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)) 
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (N,k)
	neighbours = coords[:,indices] # (3,N,k)
	keeping = distances<radius # (N,k)
	Ns = np.sum(keeping, axis = 1) # (N)
	means = np.sum(neighbours*keeping/Ns[None,:,None], axis = 2) # (3,N)
	raw_deviations = (neighbours - means[:,:,None])*keeping # (3,N,k)
	xy_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	yz_covs = np.sum(raw_deviations[1,:,:]*raw_deviations[2,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	zx_covs = np.sum(raw_deviations[2,:,:]*raw_deviations[0,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	xx_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[0,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	yy_covs = np.sum(raw_deviations[1,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	zz_covs = np.sum(raw_deviations[2,:,:]*raw_deviations[2,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	evals, evects = LA.eigh(np.swapaxes(np.dstack((np.vstack((xx_covs,xy_covs,zx_covs)),np.vstack((xy_covs,yy_covs,yz_covs)),np.vstack((zx_covs,yz_covs,zz_covs)))),0,1))
	good_pts = Ns>=m
	pt_ct = np.sum(keeping, axis = 1)
	exp1 = xx_covs*yy_covs*zz_covs+2*xy_covs*yz_covs*zx_covs
	exp2 = xx_covs*yz_covs*yz_covs+yy_covs*zx_covs*zx_covs+zz_covs*xy_covs*xy_covs
	lin_regs = abs(xy_covs*yz_covs*zx_covs/(xx_covs*yy_covs*zz_covs))
	xy_lin_regs = abs(xy_covs/np.sqrt(xx_covs*yy_covs))
	plan_regs = exp2/exp1
	lin_regs[np.logical_or(np.isnan(lin_regs),np.isinf(lin_regs))]=1
	xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs),np.isinf(xy_lin_regs))]=1
	plan_regs[np.logical_or(np.isnan(plan_regs),np.isinf(plan_regs))]=1
	rank = np.sum(evals>thresh, axis = 1)
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	C_int = int(clip)
	C_rat = int(100*(clip-int(clip)))
	T_int = int(thresh)
	T_rat = int(1000*(thresh-int(thresh)))
	points = in_file.points
	out_file0 = File(file_name[:-4]+"LinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.points = points
	out_file0.intensity = 1000*lin_regs
	out_file0.close()
	end = time.time()
	out_file1 = File(file_name[:-4]+"XYLinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file1.points = points
	out_file1.intensity = 1000*xy_lin_regs
	out_file1.close()
	out_file2 = File(file_name[:-4]+"PlanRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file2.points = points
	out_file2.intensity = 1000*plan_regs
	out_file2.close()
	out_file3 = File(file_name[:-4]+"PtCtMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file3.points = points
	out_file3.intensity = 1000*((pt_ct/k))
	out_file3.close()
	out_file4 = File(file_name[:-4]+"RuggednessMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file4.points = points
	out_file4.intensity = makesensible(1000*np.sqrt(zz_covs))
	out_file4.close()
	out_file5 = File(file_name[:-4]+"DiscXYLinRegMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+"Clip"+str(C_int)+"_"+str(C_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file5.points = in_file.points[xy_lin_regs>clip]
	out_file5.close()
	out_file6 = File(file_name[:-4]+"FewPtsMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file6.points = in_file.points[np.logical_and(pt_ct<k,good_pts)]
	out_file6.close()
	out_file7 = File(file_name[:-4]+"Eigenvalue0Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file7.points = in_file.points
	out_file7.intensity = 1000*evals[:,0]
	out_file7.close()
	out_file8 = File(file_name[:-4]+"Eigenvalue1Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file8.points = in_file.points
	out_file8.intensity = 1000*evals[:,1]
	out_file8.close()
	out_file9 = File(file_name[:-4]+"Eigenvalue2Min"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file9.points = in_file.points
	out_file9.intensity = 1000*evals[:,2]
	out_file10 = File(file_name[:-4]+"RankMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+"Thresh"+str(T_int).zfill(2)+"_"+str(T_rat).zfill(3)+".las", mode = "w", header = in_file.header)
	out_file10.points = in_file.points
	out_file10.raw_classification = rank
	out_file10.close()
	out_file11 = File(file_name[:-4]+"NNDMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+"Thresh"+str(T_int).zfill(2)+"_"+str(T_rat).zfill(3)+".las", mode = "w", header = in_file.header)
	out_file11.points = in_file.points
	out_file11.intensity = 10000*distances[:,1]
	out_file11.close()
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def attr(file_name, k= 50, radius = 0.75, thresh = 0.001,v_speed = 0):
# in comments below N is num pts in file
# parenthetic comments indicate shape of array
	start = time.time()
	in_file = File(file_name, mode = "r")
	coords = np.vstack((in_file.x,in_file.y,in_file.z,v_speed*in_file.gps_time)) # (3,N)
	distances, indices = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)).kneighbors(np.transpose(coords)) # (N,k)
	neighbours = (coords)[:,indices] # (3,N,k)
	keeping = distances<radius # (N,k)
	Ns = np.sum(keeping, axis = 1) # (N)
	#means = np.sum(neighbours*keeping/Ns[None,:,None], axis = 2) # (3,N)
	means = coords[:,:,None]
	raw_deviations = keeping*(neighbours - means)/np.sqrt(Ns[None,:,None]) # (3,N,k)
	cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(N,3,3)
	xy_covs = cov_matrices[:,0,1]
	yz_covs = cov_matrices[:,1,2]
	zx_covs = cov_matrices[:,2,0]
	xx_covs = cov_matrices[:,0,0]
	yy_covs = cov_matrices[:,1,1]
	zz_covs = cov_matrices[:,2,2]
	evals, evects = LA.eigh(cov_matrices)
	pt_ct = np.sum(distances<radius, axis = 1)
	exp1 = xx_covs*yy_covs*zz_covs+2*xy_covs*yz_covs*zx_covs
	exp2 = xx_covs*yz_covs*yz_covs+yy_covs*zx_covs*zx_covs+zz_covs*xy_covs*xy_covs
	xy_lin_regs = abs(xy_covs/np.sqrt(xx_covs*yy_covs))
	plan_regs = exp2/exp1
	xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs),np.isinf(xy_lin_regs))]=1
	plan_regs[np.logical_or(np.isnan(plan_regs),np.isinf(plan_regs))]=1
	ranks = np.sum(evals>thresh, axis = 1)
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	T_int = int(thresh)
	T_rat = int(1000*(thresh-int(thresh)))
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	Header = in_file.header
	out_file = File(file_name[:-4]+"_"+str(k).zfill(3)+"_"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+"_"+str(T_int).zfill(1)+"_"+str(T_rat).zfill(3)+"_"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2)+".las", mode = "w", header = Header)
	out_file.define_new_dimension(name = "c_xy", data_type = 9, description = "XY-covariance")
	out_file.define_new_dimension(name = "c_yz", data_type = 9, description = "YZ-covariance")
	out_file.define_new_dimension(name = "c_zx", data_type = 9, description = "ZX-covariance")
	out_file.define_new_dimension(name = "c_xx", data_type = 9, description = "XX-covariance")
	out_file.define_new_dimension(name = "c_yy", data_type = 9, description = "YY-covariance")
	out_file.define_new_dimension(name = "c_zz", data_type = 9, description = "ZZ-covariance")
	out_file.define_new_dimension(name = "xy_lin_reg", data_type = 9, description = "XY-linear regression")
	out_file.define_new_dimension(name = "plan_reg", data_type = 9, description = "Planar regression")
	out_file.define_new_dimension(name = "eig0", data_type = 9, description = "Eigenvalue 0")
	out_file.define_new_dimension(name = "eig1", data_type = 9, description = "Eigenvalue 1")
	out_file.define_new_dimension(name = "eig2", data_type = 9, description = "Eigenvalue 2")
	out_file.define_new_dimension(name = "rank", data_type = 5, description = "SVD rank")
	for dimension in in_file.point_format:
		dat = in_file.reader.get_dimension(dimension.name)
		out_file.writer.set_dimension(dimension.name, dat)
	out_file.c_xy = xy_covs
	out_file.c_yz = yz_covs
	out_file.c_zx = zx_covs
	out_file.c_xx = xx_covs
	out_file.c_yy = yy_covs
	out_file.c_zz = zz_covs
	out_file.xy_lin_reg = xy_lin_regs
	out_file.plan_reg = plan_regs
	out_file.eig0 = evals[:,0]
	out_file.eig1 = evals[:,1]
	out_file.eig2 = evals[:,2]
	out_file.classification = ranks
	out_file.close()
	end = time.time()
	print(file_name, "done. Time taken = "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

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

def intsdens(file_name, m = 4, k= 50, radius = 0.75):
# in comments below N is num pts in file
# parenthetic comments indicate shape of array
	start = time.time()
	in_file = File(file_name+".las", mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array)) # (2,N)
	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)) 
	print("Training done...")
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (N,k)
	print("Distances and indices found...")
	intensities = z_array[indices] # (N,k)
	keeping = distances<radius # (N,k)
	Ns = np.sum(keeping, axis = 1) # (N)
	inden = np.max(intensities, axis = 1) # (N)
	end = time.time()
	print("Time taken so far: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
	evals = inden
	good_pts = Ns>=m
	print("Calculations done... making LAS files")
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	points = in_file.points
	out_file0 = File(file_name+"IntDensMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.points = points
	out_file0.intensity = makesensible(inden)
	out_file0.close()
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

# T A K E S   P O I N T  D E N S I T Y
# K E E PS   O N LY   P O I N T S  W I T H  P T  D E N S I T Y   G R E A TE R   T H A N  m
def ptdens(file_name, radius = 1):
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
	out_file = File(file_name+"PtDensityRadius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points
	out_file.intensity = makesensible(pt_ct/radius**3)
	out_file.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

# N E A R   F L I G H T   L I N E
def nfl(file_name, clip = 100, change_name = True):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	class10 = in_file.classification==10
	x_array = in_file.x
	y_array = in_file.y
	coords = np.vstack((x_array,y_array))
	x_flight = x_array[class10]
	y_flight = y_array[class10]
	coords_flight = np.vstack((x_flight,y_flight))
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(coords_flight))
	distances, indices = nhbrs.kneighbors(np.transpose(coords))
	out_file = File(file_name[:-4]+change_name*("NFLClip"+str(int(clip)).zfill(3)+"_"+str(int(100*(clip-int(clip)))).zfill(2))+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points[np.logical_and(distances[:,0]<clip, np.logical_not(class10))]
	out_file.close()
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

# W E I G H T S   F R O M   K - N E I G H B O U R S   G R A P H
def nng(file_name, k=1, components = 10):
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
	out_file.intensity = makesensible(sizes)
	out_file.close()
	out_file1 = File(file_name+"Components"+str(components).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file1.points = in_file.points
	out_file1.intensity = makesensible(np.bincount(labels)[labels])
	out_file1.close()
	end = time.time()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def maxrets(file_name, m = 4, k= 50, radius = 0.75):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array)) # (3,N)
	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)) 
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (N,k)
	neighbours = coords[:,indices] # (3,N,k)
	keeping = distances<radius # (N,k)
	maxrets = np.max(in_file.num_returns[indices], axis = 1)
	out_file0 = File(file_name[:-4]+"MaxRetsMin"+str(m).zfill(3)+"Max"+str(k).zfill(3)+"Radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+".las", mode = "w", header = in_file.header)
	out_file0.points = points
	out_file0.raw_classification = maxrets
	out_file0.close()
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def extractconductor(file_name, k=50, m = 4, radius = 0.75, clip = 0.99):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coords = np.vstack((x_array,y_array,z_array))
	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (N,k)
	neighbours = coords[:,indices] # (3,N,k)
	keeping = distances<radius # (N,k)
	Ns = np.sum(keeping, axis = 1) # (N)
	means = np.sum(neighbours*keeping/Ns[None,:,None], axis = 2) # (3,N)
	raw_deviations = (neighbours - means[:,:,None])*keeping # (3,N,k)
	xy_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	xx_covs = np.sum(raw_deviations[0,:,:]*raw_deviations[0,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	yy_covs = np.sum(raw_deviations[1,:,:]*raw_deviations[1,:,:]*keeping/Ns[:,None], axis = 1) # (N)
	good_pts = Ns>=m
	xy_lin_regs = abs(xy_covs/np.sqrt(xx_covs*yy_covs))
	xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs),np.isinf(xy_lin_regs))]=1
	x_array = x_array
	y_array = y_array
	z_array = z_array
	coords = np.vstack((x_array,y_array, z_array)) # (2,N)
	tree = KDTree(np.transpose(coords), leaf_size=2)
	pt_ct = tree.query_radius(np.transpose(coords), r=radius, count_only=True)
	out_file = File("ExractConductor"+file_name, mode = "w", header = in_file.header)
	out_file.points = in_file.points[xy_lin_reg>clip]
	out_file.intensity = pt_ct[xy_lin_regs>clip]
	out_file.close()
	print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
