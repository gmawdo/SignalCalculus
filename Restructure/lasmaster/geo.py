import numpy as np
from sklearn.neighbors import NearestNeighbors
from lasmaster.infotheory import entropy

def ladder_tensor(m, n):
	I = np.empty((n,n,n))
	for i in range(n):
		I[i,:,:]=np.identity(n)
		I[i,(i+1):,:]=0
	return I[m-1:,:,:] # returns an (n-m+1) x n x n, rank 3 tensor


def optimise_k(coords, distances, indices, min_num, k, optimise):
	stack = ()
	if optimise:
		aux = k
	else:
		aux = min_num
	for j in range(aux, k+1):
		indicesj = indices[:,:j]
		distancesj = distances[:,:j]
		raw_deviations = ((coords)[:,indicesj] - coords[:,:,None])/np.sqrt(j) # (d,num_pts,k)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		# the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
		# this is crucial in order to get accurate eigenvalues and eigenvectors
		cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
		evals, evects = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)
		linearity = (evals[:,-1]-evals[:,-2])/evals[:,-1]
		planarity = (evals[:,-2]-evals[:,-3])/evals[:,-1]
		scattering = evals[:,-3]/evals[:,-1]
		dim_ent = entropy(np.stack((linearity, planarity, scattering), axis = 1))
		stack = stack+(dim_ent,)
	dimensional_entropy = np.stack(stack, axis = 1)
	k_opt = aux+np.argmin(dimensional_entropy, axis = 1)
	return k_opt, evals, evects

def optimise_radius(coord_dictionary, config):
	x = coord_dictionary["x"]
	y = coord_dictionary["y"]
	z = coord_dictionary["z"]
	t = coord_dictionary["gps_time"]
	
	min_num = 4

	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimation"]
	optimise = config["k-optimise"]

	spacetime = bool(v_speed)
	decimate = bool(u)
	coords = np.vstack((x,y,z)+spacetime*(v_speed*t,))
	# work out how many dimensions we are working in
	d = 3+spacetime

	stack = ()
	
	if optimise:
		aux = min_num
	else:
		aux = k

	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (num_pts,k)
	J = ladder_tensor(aux, k) # we use this tensor to avoid a loop # (k-aux+1, k, k)
	raw_deviations_prestack = (coords[:,indices] - coords[:,:,None]) # (d,num_pts,k)
	raw_deviations = np.matmul(raw_deviations_prestack[:,None,:,:], J) # (d,k-aux+1,num_pts,k)
	cov_matrices = np.matmul(raw_deviations.transpose(1,2,0,3), raw_deviations.transpose(1,2,3,0)) #(num_pts,d,d)
	for j in range(aux, k+1):
		indicesj = indices[:,:j]
		distancesj = distances[:,:j]
		coordsj = (coords)[:,indicesj]
		raw_deviations = (coordsj - coords[:,:,None])/np.sqrt(j) # (d,num_pts,k)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		# the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
		# this is crucial in order to get accurate eigenvalues and eigenvectors
		cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
		evals, evects = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)
		linearity = (evals[:,-1]-evals[:,-2])/evals[:,-1]
		planarity = (evals[:,-2]-evals[:,-3])/evals[:,-1]
		scattering = evals[:,-3]/evals[:,-1]
		dim_ent = entropy(np.stack((linearity, planarity, scattering), axis = 1))
		stack = stack+(dim_ent,)
		print(str(j), "/", str(k))
	dimensional_entropy = np.stack(stack, axis = 1)
	k_opt = aux + np.argmin(dimensional_entropy, axis = 1)
	print(min(k_opt), max(k_opt))
	#radius_opt = distances[np.arange(distances.shape[0]),k_opt-1]
	#radius_opt = np.median(radius_opt[indices], axis = 1)
	#radius_opt = np.median(radius_opt[indices], axis = 1)
	return k_opt

def optimise_k(coord_dictionary, config):
	x = coord_dictionary["x"]
	y = coord_dictionary["y"]
	z = coord_dictionary["z"]
	t = coord_dictionary["gps_time"]
	
	min_num = 4

	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimation"]
	optimise = config["k-optimise"]

	spacetime = bool(v_speed)
	decimate = bool(u)
	coords = np.vstack((x,y,z)+spacetime*(v_speed*t,))
	# work out how many dimensions we are working in
	d = 3+spacetime

	stack = ()
	
	if optimise:
		aux = min_num
	else:
		aux = k

	nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
	distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (num_pts,k)
	for j in range(aux, k+1):
		indicesj = indices[:,:j]
		distancesj = distances[:,:j]
		coordsj = (coords)[:,indicesj]
		raw_deviations = (coordsj - coords[:,:,None])/np.sqrt(j) # (d,num_pts,k)
		cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
		# the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
		# this is crucial in order to get accurate eigenvalues and eigenvectors
		cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
		evals, evects = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)
		linearity = (evals[:,-1]-evals[:,-2])/evals[:,-1]
		planarity = (evals[:,-2]-evals[:,-3])/evals[:,-1]
		scattering = evals[:,-3]/evals[:,-1]
		dim_ent = entropy(np.stack((linearity, planarity, scattering), axis = 1))
		stack = stack+(dim_ent,)
		print(str(j), "/", str(k))
	dimensional_entropy = np.stack(stack, axis = 1)
	k_opt = aux + np.argmin(dimensional_entropy, axis = 1)
	print(min(k_opt), max(k_opt))
	#radius_opt = distances[np.arange(distances.shape[0]),k_opt-1]
	#radius_opt = np.median(radius_opt[indices], axis = 1)
	#radius_opt = np.median(radius_opt[indices], axis = 1)
	return k_opt


# G E O M E T R Y   M O D U L E
# takes a dictionary which contains x, y, z and time components 
# and returns the eigeninformation, and kdistance
# that is, this function extracts local geometric information
def eig(coord_dictionary, config):
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

	# work out how many dimensions we are working in
	d = 3+spacetime
	
	# find time bins

	times = [np.quantile(t, q=i/N) for i in range(N+1)]

	# generate labels for new attributes
	val1 = np.empty(t.shape)
	val2 = np.empty(t.shape)
	val3 = np.empty(t.shape)
	vec1 = np.empty(t.shape+(3,))
	vec2 = np.empty(t.shape+(3,))
	vec3 = np.empty(t.shape+(3,))
	kdist = np.empty(t.shape)

	# loop around time ranges
	for i in range(N):

		# work out which time range we are working in
		time_range = (times[i]<=t)*(t<=times[i+1])

		coords = np.vstack((x[time_range],y[time_range],z[time_range])+spacetime*(v_speed*t[time_range],))
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
	return val1, val2, val3, vec1, vec2, vec3, k, kdist #note that I have not set k or kdist properly