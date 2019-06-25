import numpy as np
from sklearn.neighbors import NearestNeighbors
from lasmaster.infotheory import entropy
from lasmaster import fun

def read_config(coord_dictionary, config):
	import time
	start = time.time()
	
	x = coord_dictionary["x"]
	y = coord_dictionary["y"]
	z = coord_dictionary["z"]
	t = coord_dictionary["gps_time"]
	
	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimation"]

	spacetime = bool(v_speed)
	decimate = bool(u)

	times = [np.quantile(t, q = r) for r in np.linspace(0,1,N+1)]
	time_digits = np.digitize(t, bins = times, right = False) # bins 0,1,2,...,N+1. 0 represents less than min, N+1 more than max.
	time_digits[time_digits<1]=1
	time_digits[time_digits>N]=N
	# work out how many dimensions we are working in
	
	k_opt = np.zeros(t.size, dtype = int)
	# find time bins
	for i in np.unique(time_digits):

		# work out which time range we are working in
		time_range = time_digits == i

		coords = np.vstack((x[time_range],y[time_range],z[time_range])+spacetime*(v_speed*t[time_range],))
		d = coords.shape[0]
		# num_pts = coords.shape[-1] denotes the number of working points in comments
		if decimate:
			spatial_coords, ind, inv, cnt = np.unique(np.floor(coords[0:d,:]/u), return_index = True, return_inverse = True, return_counts = True, axis=1)
		else:
			ind = np.arange(sum(time_range))
			inv = ind
		coords = coords[:,ind] # (d,num_pts)

		# find the k nearest neighbours of each point
		nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
		distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (num_pts,k)
		# (d,num_pts,k)
		k_opt[time_range] = optimise_k(coords, distances, indices)
		# work out which neighbours are being disregarded due to distance
		#keeping = distances < radius # (num_pts,k)
		#ks = np.sum(keeping, axis = 1) # (num_pts)


		end = time.time()
		print("Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

	return k_opt


# O P T I M I S E   K   N U M B E R S
def optimise_k(coords, distances, indices):
	d = coords.shape[0]
	num_pts = indices.shape[0] #coords has shape (d,num_points)
	k = indices.shape[1]
	optimise = True
	
	k_ran = np.intersect1d(np.arange(100),np.arange(3,k))
	relative_positions = coords[:,indices] - coords[:,:,None] #(d,num_pts,k)
	print(indices.shape, relative_positions.shape)
	

	k_opt = np.zeros(num_pts)
	entropy_store = np.ones(num_pts, dtype = float)



	for index, item in enumerate(k_ran):
		# matrix multiplications
		raw_deviations = relative_positions[:, :, :item + 1] #(d,num_pts,item + 1)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1)) #(num_pts,d,d)
		# get eigeninformation
		print(item)
		evals, evects = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)
		linearity = (evals[:,-1]-evals[:,-2])/evals[:,-1] #num_pts
		planarity = (evals[:,-2]-evals[:,-3])/evals[:,-1] #num_pts
		scattering = evals[:,-3]/evals[:,-1] #num_pts
		dim_ent = entropy(np.stack((linearity, planarity, scattering), axis = 1)) #(num_pts)
		k_opt[dim_ent<entropy_store] = item #num_pts
		entropy_store[dim_ent<=entropy_store] = dim_ent[dim_ent<entropy_store]

	return k_opt

# E I G E N I N F O
# takes a dictionary which contains x, y, z and time components 
# and returns the eigeninformation, and kdistance
# that is, this function extracts local geometric information
def eig(coord_dictionary, config):
	x = coord_dictionary["x"]
	y = coord_dictionary["y"]
	z = coord_dictionary["z"]
	t = coord_dictionary["gps_time"]

	N = config["timeIntervals"]
	k_range = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	optimise = config["k-optimise"]

	spacetime = bool(v_speed)

	# work out how many dimensions we are working in
	d = 3+int(spacetime)

	# find time bins

	times = [np.quantile(t, q = r) for r in np.linspace(0,1,N+1)]
	time_digits = np.digitize(t, bins = times, right = False) # bins 0,1,2,...,N+1. 0 represents less than min, N+1 more than max.
	time_digits[time_digits<1]=1
	time_digits[time_digits>N]=N
	# work out how many dimensions we are working in

	# generate labels for new attributes
	val1 = np.empty(t.shape, dtype = float)
	val2 = np.empty(t.shape, dtype = float)
	val3 = np.empty(t.shape, dtype = float)
	vec1 = np.empty(t.shape+(3,), dtype = float)
	vec2 = np.empty(t.shape+(3,), dtype = float)
	vec3 = np.empty(t.shape+(3,), dtype = float)
	kdist = np.empty(t.shape, dtype = float)
	kopt = np.empty(t.shape, dtype = int)
	
	coords = np.vstack((x,y,z)+spacetime*(v_speed*t))
	nhbrs = NearestNeighbors(n_neighbors = max(k_range), algorithm = "kd_tree").fit(np.transpose(coords))
	# loop around time ranges
	for i in np.unique(time_digits):

		# work out which time range we are working in
		time_range = time_digits == i

		# find the k nearest neighbours of each point
		distances, indices = nhbrs.kneighbors(np.transpose(coords[time_range,:])) # (num_pts,k)
		unq, ind, inv = np.unique(np.ravel(indices), return_index = True, return_inverse = True)
		coords_for_kopt = coords[:,unq]
		indices_for_kopt = inv[ind].reshape
		# work out which neighbours are being disregarded due to distance
		keeping = distances < radius # (num_pts,k)
		ks = np.sum(keeping, axis = 1) # (num_pts)

		#raw_deviations = keeping[k_opt == j,:j]*(coords[:,indices[k_opt == j,:j]] - coords[:, k_opt == j,None])/np.sqrt(ks[None,k_opt == j,None]) # (d,num_pts,k)
		#cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		## the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
		## this is crucial in order to get accurate eigenvalues and eigenvectors
		#cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
		#evals[k_opt == j], evects[k_opt == j] = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)

		raw_deviations = keeping*(coords[:,indices] - coords[:,:,None])/np.sqrt(ks[None,:,None]) # (d,num_pts,k)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		# the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
		# this is crucial in order to get accurate eigenvalues and eigenvectors
		cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
		evals, evects = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)

		val1[time_range] = evals[inv,-3]
		val2[time_range] = evals[inv,-2]
		val3[time_range] = evals[inv,-1]
		vec1[time_range,:] = evects[inv,:-1,-3]/(np.linalg.norm(evects[inv,:-1,-3], axis = 1)[:,None])
		vec2[time_range,:] = evects[inv,:-1,-2]/(np.linalg.norm(evects[inv,:-1,-2], axis = 1)[:,None])
		vec3[time_range,:] = evects[inv,:-1,-1]/(np.linalg.norm(evects[inv,:-1,-1], axis = 1)[:,None])
		kdist[time_range] = distances[inv,-1]
	return val1, val2, val3, vec1, vec2, vec3, kopt, kdist #note that I have not set k or kdist properly