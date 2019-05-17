# L A S M A S T E R

import numpy as np
import time
from laspy.file import File
from sklearn.neighbors import NearestNeighbors

# E N T R O P Y   O F   P  R O B A B I L I T Y   V E C T O R
def entropy(distribution):
	presum = -distribution*np.log(distribution)/np.log(2)
	presum[np.logical_or(np.isnan(presum),np.isinf(presum))]=0
	return np.sum(presum)

# J E N S E N - S H A N N O N   D I V E R G E N C E
def jsd(dist1,dist2):
	average = 0.5*(dist1+dist2)
	return entropy(average)-0.5*(entropy(dist1)+entropy(dist2))

# A T T R I B U T E S
def attr(file_name, N=6, k = 50, radius = 0.5, thresh = 0.001, spacetime = True, v_speed = 2, decimate = True, u = 0.1):
	# in comments below N is num pts in file
	# parenthetic comments indicate shape of array, d = 4 when spacetime = True, d = 3 otherwise
	start = time.time() 

	in_file = File(file_name, mode = "r") # Reads in the file with laspy

	if v_speed == 0: #v_speed is a parameter which is related to the "spacetime catastrophe" solution
		spacetime = False
	
	d = 3+spacetime
	
	# find time bins
	times = list([np.quantile(in_file.gps_time, q=i/N) for i in range(N+1)])

	# create some parameters for naming the output
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	T_int = int(thresh)
	T_rat = int(1000*(thresh-int(thresh)))
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	U_int = int(u)
	U_rat = int(100*(u-int(u)))
	K = "k"+str(k).zfill(3)
	R = "radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)
	T = "thresh"+str(T_int).zfill(1)+"_"+str(T_rat).zfill(3)
	C = "vspeed"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2)
	U = decimate*("dec"+str(U_int).zfill(2)+"_"+str(U_rat).zfill(2))
	out_name = "attr"+file_name[:-4]+K+R+T+C+U+".las"
	
	#create new file
	Header = in_file.header
	out_file = File(out_name, mode = "w", header = Header)
	
	# create a dictionary to store the new attribute names
	continuous_attributes = 	{
					"xy_lin_reg"	:	"XY-linear regression",
					"plan_reg"	:	"Planar regression",
					"eig0"		:	"Eigenvector 0",
					"eig1"		:	"Eigenvector 1",
					"eig2"		:	"Eigenvector 2",
					"lin_reg"	:	"Linear regression",
					"curv"		:	"Normal curvature",
					"iso"		:	"Isotropy",
					"ent"		:	"Entropy",
					"plang"		:	"Planar regression",
					"lang"		:	"Linear angle",
					"impdec"	:	"Point density",
					}
	discrete_attributes = 		{
					"rank"		:	"SVD rank",
					}
	
	# extract names of pre-existing attributes
	dimensions = [spec.name for spec in in_file.point_format]

	# create new dimensions	
	for dimension in continuous_attributes:
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = 9, description = continuous_attributes[dimension])
	for dimension in discrete_attributes:
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = 5, description = discrete_attributes[dimension])

	# add pre-existing point records
	for dimension in dimensions:
		dat = in_file.reader.get_dimension(dimension)
		out_file.writer.set_dimension(dimension, dat)
	
	#create a dictionary to store new attributes
	attributes = {}
	for dimension in continuous_attributes:
		attributes[dimension]=out_file.reader.get_dimension(dimension)
	for dimension in discrete_attributes:
		attributes[dimension]=out_file.reader.get_dimension(dimension)

	# loop around time ranges
	for i in range(N):
		time_range = (times[i]<=in_file.gps_time)*(in_file.gps_time<=times[i+1])

		# do the maths
		coords = np.vstack((in_file.x[time_range],in_file.y[time_range],in_file.z[time_range])+spacetime*(v_speed*in_file.gps_time[time_range],)) # (d,N)
		if decimate:
			spatial_coords, ind, inv, cnt = np.unique(np.floor(coords[0:d,:]/u), return_index = True, return_inverse = True, return_counts = True, axis=1)
		else:
			ind = np.arange(len(in_file.x[time_range]))
			inv = ind
		coords = coords[:,ind]
		distances, indices = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords)).kneighbors(np.transpose(coords)) # (N,k)
		neighbours = (coords)[:,indices] # (d,N,k)
		keeping = distances<radius # (N,k)
		Ns = np.sum(keeping, axis = 1) # (N)
		means = coords[:,:,None]
		raw_deviations = keeping*(neighbours - means)/np.sqrt(Ns[None,:,None]) # (d,N,k)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(N,d,d)
		xy_covs = cov_matrices[:,0,1]
		yz_covs = cov_matrices[:,1,2]
		zx_covs = cov_matrices[:,2,0]
		xx_covs = cov_matrices[:,0,0]
		yy_covs = cov_matrices[:,1,1]
		zz_covs = cov_matrices[:,2,2]
		evals, evects = np.linalg.eigh(cov_matrices)
		evals2, evects2 = np.linalg.eigh(cov_matrices[:,0:2,0:2])
		exp1 = xx_covs*yy_covs*zz_covs+2*xy_covs*yz_covs*zx_covs
		exp2 = xx_covs*yz_covs*yz_covs+yy_covs*zx_covs*zx_covs+zz_covs*xy_covs*xy_covs
		xy_lin_regs = abs(xy_covs/np.sqrt(xx_covs*yy_covs))
		plan_regs = exp2/exp1
		lin_regs = abs(xy_covs*yz_covs*zx_covs/(xx_covs*yy_covs*zz_covs))
		xy_lin_regs[np.logical_or(np.isnan(xy_lin_regs),np.isinf(xy_lin_regs))]=1
		plan_regs[np.logical_or(np.isnan(plan_regs),np.isinf(plan_regs))]=1
		lin_regs[np.logical_or(np.isnan(lin_regs),np.isinf(lin_regs))]=1
		ranks = np.sum(evals>thresh, axis = 1)
		means = np.mean(neighbours, axis = 2)
		p0 = evals[:,-3]/(evals[:,-1]+evals[:,-2]+evals[:,-3])
		p1= evals[:,-2]/(evals[:,-1]+evals[:,-2]+evals[:,-3])
		p2 = evals[:,-1]/(evals[:,-1]+evals[:,-2]+evals[:,-3])
		p1 = -p1*np.log(p1)
		p2 = -p2*np.log(p2)
		p0[np.isnan(p0)]=0
		p1[np.isnan(p1)]=0
		p2[np.isnan(p2)]=0
		E = (p0+p1+p2)/np.log(3)
		if not(decimate):
			dens = 3*k/(4*np.pi*(distances[:,-1]**3))
		else:
			dens = cnt/(u**3)
		isos = (evals[:,-1]+evals[:,-2]+evals[:,-3])/np.sqrt(3*((evals[:,-1]**2+evals[:,-2]**2+evals[:,-3]**2)))
		plangs = np.clip(2*(np.arccos(abs(evects[:,2,-3])/(np.sqrt(evects[:,2,-3]**2+evects[:,1,-3]**2+evects[:,0,-3]**2)))/np.pi),0,1)
		langs = np.clip(2*(np.arccos(abs(evects[:,2,-1])/(np.sqrt(evects[:,2,-1]**2+evects[:,1,-1]**2+evects[:,0,-1]**2)))/np.pi),0,1)

		# update attribute values
		attributes["xy_lin_reg"][time_range] = xy_lin_regs[inv]
		attributes["lin_reg"][time_range] = lin_regs[inv]
		attributes["plan_reg"][time_range] = plan_regs[inv]
		attributes["eig0"][time_range] = evals[:,-3][inv]
		attributes["eig1"][time_range] = evals[:,-2][inv]
		attributes["eig2"][time_range] = evals[:,-1][inv]
		attributes["curv"][time_range] = p0[inv]
		attributes["iso"][time_range] = isos[inv]
		attributes["rank"][time_range] = ranks[inv]
		attributes["impdec"][time_range] = dens[inv]
		attributes["ent"][time_range] = E[inv]
		attributes["plang"][time_range] = plangs[inv]
		attributes["lang"][time_range] = langs[inv]

	# check for nans or infs
	for dimension in continuous_attributes:
		attributes[dimension][np.logical_or(np.isnan(attributes[dimension]),np.isinf(attributes[dimension]))]=0

	out_file.close()
	end = time.time()
	print(file_name, "done. Time taken = "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

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
	return file_name[:-4]+change_name*("NFLClip"+str(int(clip)).zfill(3)+"_"+str(int(100*(clip-int(clip)))).zfill(2))+".las"
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")




# takes a file and an example file and gives a plot of how well the file is correlated with the example
def correlator(in_file, attributes, example, name, N=20, L=3):
	unq, ind, inv, cnt = np.unique(np.floor(np.vstack((in_file.x,in_file.y,in_file.z))/L), return_index = True, return_inverse = True, return_counts = True, axis=1)
	intensity = 0*in_file.intensity

	# get attributes from example data set
	example_attributes = {}
	for dimension in attributes:
		example_attributes[dimension]=example.reader.get_dimension(dimension)	

	# make the histogram "wavelet" from the data set
	example_digits = {}
	for dimension in attributes:
		example_digits[dimension] = np.digitize(example_attributes[dimension], bins = np.linspace(0,1,N+1))-1
		print(example_digits[dimension])
	example_ids = sum(((N**index)*example_digits[dimension] for index, dimension in enumerate(attributes)))
	print(example_ids)
	example_histogram, example_edges = np.histogram(example_ids, bins = range(N**len(attributes)), density = True)

	# find JSD from "wavelet" to patches of "signal"
	for entry in np.unique(inv):
		voxel_attributes = {}
		for dimension in attributes:
			voxel_attributes[dimension]=in_file.reader.get_dimension(dimension)[inv == entry]
		
		voxel_digits = {}
		for dimension in attributes:
			voxel_digits[dimension] = np.digitize(voxel_attributes[dimension], bins = np.linspace(0,1,N+1))-1
		
		voxel_ids = sum(((N**index)*voxel_digits[dimension] for index, dimension in enumerate(attributes)))
		voxel_histogram, voxel_edges = np.histogram(voxel_ids, bins = range(N**len(attributes)), density = True)
		
		intensity[inv == entry]=1000*(1-jsd(example_histogram, voxel_histogram))
		
	outFile = File(name+".las", mode = "w", header = in_file.header)
	outFile.points = in_file.points
	outFile.intensity = intensity

	outFile.close()
	


