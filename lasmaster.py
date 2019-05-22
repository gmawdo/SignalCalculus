# L A S M A S T E R

import numpy as np
import time
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# U S E   L A S P Y   T O   T U R N   A   L A S   F I L E   I N T O
# A   D I C T I O N A R Y   O F   A T T R I B U T E S
# A N D   A   H E A D E R
def read_las(file_name):
	in_file = File(file_name, mode = "r")
	dimensions = [spec.name for spec in in_file.point_format]
	attributes = {}
	for dimension in attributes:
		dat = in_file.reader.get_dimension(dimension)
		attributes[dimension] = dat
	return (in_file.header, attributes)

# W R I T I N G   D A T A   T O   L A S
def write_las(name, header, attributes, data_types, descriptions):
	# create new dimensions	
	out_file = File(name, mode = "w", header = header)
	dimensions = [spec.name for spec in out_file.point_format]
	for dimension in attributes:
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = data_types[dimension], description = descriptions[dimension])

	# populate point records
	for dimension in attributes:
		dat = attributes[dimension]
		out_file.writer.set_dimension(dimension, dat)
	

# E N T R O P Y   O F   P R O B A B I L I T Y   V E C T O R S
# input: (...,N) probability vectors
# that is, we must have np.sum(distribution, axis = -1)=1
# output: entropies of vectors
def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
	return entropies

# J E N S E N - S H A N N O N   D I V E R G E N C E
# input: (...,M,N) matrices whose rows are probability vectors
# output: J-S div. of the collection of vectors
def jsd(distribution):
	M = distribution.shape[-2]
	N = distribution.shape[-1]
	return (entropy(np.mean(distribution, axis = -2))-np.mean(entropy(distribution), axis = -1))*np.log(N)/np.log(M)

# F U N C T I O N S   OF   E I G E N V A L U E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvalues
def fun_eig(thresh):
	output =	{
			"p0"	:	(lambda x, y, z: x/(x+y+z)),
			"p1"	:	(lambda x, y, z: y/(x+y+z)),
			"p2"	:	(lambda x, y, z: z/(x+y+z)),
			"eig0"	:	(lambda x, y, z: x),
			"eig1"	:	(lambda x, y, z: y),
			"eig2"	:	(lambda x, y, z: z),
			"iso"	:	(lambda x, y, z: (x+y+z)/np.sqrt(3*(x**2+y**2+z**2))),
			"ent"	:	(lambda x, y, z: entropy(np.stack((x, y, z))/(x+y+z))),
			"rank"	:	(lambda x, y, z: 1*(x>thresh)+1*(y>thresh)+1*(z>thresh)),
			}
	return output

# F U N C T I O N S   O F   E I G E N V E C T O R S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvectors
def fun_vec():
	output =	{					
			"ang0"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v0[:,2])/(np.sqrt(v0[:,0]**2+v0[:,1]**2+v0[:,2]**2)))/np.pi),0,1)),
			"ang1"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v1[:,2])/(np.sqrt(v1[:,0]**2+v1[:,1]**2+v1[:,2]**2)))/np.pi),0,1)),
			"ang2"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v2[:,2])/(np.sqrt(v2[:,0]**2+v2[:,1]**2+v2[:,2]**2)))/np.pi),0,1)),
			}
	return output

# F U N C T I O N S   O F   D I S T A N C E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of k-distances
# when we decimate, these are functions of point counts and decimation parameter u
def fun_dist(decimate):
	sphere_constant = (4/3)*np.pi
	if decimate:
		output =	{
				"ptdens":	lambda k, d: k/(d**3)
				}
	else:
		output = 	{
				"ptdens":	lambda k, d: k/(sphere_constant*(d**3))
				}
	return output


# A T T R I B U T E S
def attr(file_name, N=6, k = 50, radius = 0.5, thresh = 0.001, spacetime = True, v_speed = 2, decimate = True, u = 0.1):
	# in comments below N is num pts in file, not the number of ways the tile has been split up
	# parenthetic comments indicate shape of array, d = 4 when spacetime = True, d = 3 otherwise
	start = time.time() 

	in_file = File(file_name, mode = "r") # Reads in the file with laspy

	if v_speed == 0: #v_speed is a parameter which is related to the "spacetime catastrophe" solution
		spacetime = False

	if u == 0: #v_speed is a parameter which is related to the "spacetime catastrophe" solution
		decimate = False
	
	d = 3+spacetime
	
	# find time bins
	times = [np.quantile(in_file.gps_time, q=i/N) for i in range(N+1)]


	# create some parameters for naming the output
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	T_int = int(thresh)
	T_rat = int(1000*(thresh-int(thresh)))
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	U_int = int(u)
	U_rat = int(100*(u-int(u)))
	num = "N"+str(N).zfill(3)
	K = "k"+str(k).zfill(3)
	R = "radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)
	T = "thresh"+str(T_int).zfill(1)+"_"+str(T_rat).zfill(3)
	C = "v_speed"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2)
	U = decimate*("dec"+str(U_int).zfill(2)+"_"+str(U_rat).zfill(2))
	out_name = "attr"+file_name[:-4]+num+K+R+T+C+U+".las"
	
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
	
	for dimension in attributes:
		out_file.writer.set_dimension(dimension, attributes[dimension])

	out_file.close()
	end = time.time()
	print(file_name, "done. Time taken = "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

# C L I P   N E A R   F L I G H T   L I N E
def nfl(file_name, clip = 100, fl = 10, change_name = True):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	class10 = in_file.classification==fl
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
def correlator(in_file, attributes, example, name, N=20, L=3, plot = False):
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

	example_ids = sum(((N**index)*example_digits[dimension] for index, dimension in enumerate(attributes)))
	example_histogram, example_edges = np.histogram(example_ids, bins = range(N**len(attributes)), density = True)
	
	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

	# find JSD from "wavelet" to patches of "signal"
	m = len(np.unique(inv))
	for entry in np.unique(inv):
		voxel_attributes = {}
		for dimension in attributes:
			voxel_attributes[dimension]=in_file.reader.get_dimension(dimension)[inv == entry]
			
		
		voxel_digits = {}
		for dimension in attributes:
			voxel_digits[dimension] = np.minimum(np.digitize(voxel_attributes[dimension], bins = np.linspace(0,1,N+1)),N)-1
		
		voxel_ids = sum(((N**index)*voxel_digits[dimension] for index, dimension in enumerate(attributes)))
		voxel_histogram, voxel_edges = np.histogram(voxel_ids, bins = range(N**len(attributes)), density = True)
		intensity[inv == entry]=1000*(1-jsd(np.stack((voxel_histogram, example_histogram))))
		if plot:
			xs = voxel_edges[:-1]
			ys = voxel_histogram
			zs = entry
			ax.bar(xs, ys, zs, zdir='y', alpha=0.8)
		print(entry*100/m)


	outFile = File(name, mode = "w", header = in_file.header)
	outFile.points = in_file.points
	outFile.intensity = intensity

	if plot:
		ax.set_xlabel('Attribute_Digits')
		ax.set_ylabel('Voxel')
		ax.set_zlabel('Probability')
		plt.show()
	outFile.close()


