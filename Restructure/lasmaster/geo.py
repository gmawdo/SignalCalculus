import numpy as np
from sklearn.neighbors import NearestNeighbors
from lasmaster.infotheory import entropy

def optimise_k(coords, distances, indices, min_num, k):
	stack = ()
	for j in range(min_num, k):
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
		print(j,"/",k)
	dimensional_entropy = np.stack(stack, axis = 1)
	k_opt = np.argmin(dimensional_entropy, axis = 1)
	print(min(k_opt), max(k_opt))
	print(evals.shape, evects.shape)
	return k_opt, evals, evects




# G E O M E T R Y   M O D U L E
# takes a dictionary which contains x, y, z and time components 
# and returns the eigeninformation, and kdistance
# that is, this function extracts local geometric information
def geo(coord_dictionary, config):
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

	d = 3+spacetime
	
	# find time bins

	times = [np.quantile(t, q=i/N) for i in range(N+1)]
	val1 = np.empty(t.shape)
	val2 = np.empty(t.shape)
	val3 = np.empty(t.shape)
	vec1 = np.empty(t.shape+(3,))
	vec2 = np.empty(t.shape+(3,))
	vec3 = np.empty(t.shape+(3,))
	kdist = np.empty(t.shape)
	# loop around time ranges
	for i in range(N):
		time_range = (times[i]<=t)*(t<=times[i+1])

		# do the maths
		coords = np.vstack((x[time_range],y[time_range],z[time_range])+spacetime*(v_speed*t[time_range],))
		num_pts = coords.shape[-1]
		if decimate:
			spatial_coords, ind, inv, cnt = np.unique(np.floor(coords[0:d,:]/u), return_index = True, return_inverse = True, return_counts = True, axis=1)
		else:
			ind = np.arange(sum(time_range))
			inv = ind
		coords = coords[:,ind] # (d,num_pts)
		nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
		distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (num_pts,k)
		# (d,num_pts,k)

		#Now optimise k

		keeping = distances < radius # (num_pts,k)
		ks = np.sum(keeping, axis = 1) # (num_pts)

		k_opt, evals, evects = optimise_k(coords, distances, indices, 10, k)

		for j in np.unique(k_opt):
			raw_deviations = keeping*((coords)[:,indices[k_opt == j,j]] - coords[:, k_opt == j,None])/np.sqrt(ks[None,k_opt == j,None]) # (d,num_pts,k)
			cov_matrices = np.matmul(raw_deviations.transpose(1,0,2), raw_deviations.transpose(1,2,0)) #(num_pts,d,d)
			# the next line forces cov_matrices to be symmetric so that the LAPACK routine in linalg.eigh is more stable
			# this is crucial in order to get accurate eigenvalues and eigenvectors
			cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,2,1))
			evals[k_opt == j], evects[k_opt == j] = np.linalg.eigh(cov_matrices) #(num_pts, d), (num_pts,d,d)

		val1[time_range] = evals[inv,-3]
		val2[time_range] = evals[inv,-2]
		val3[time_range] = evals[inv,-1]
		vec1[time_range,:] = evects[inv,:-1,-3]/(np.linalg.norm(evects[inv,:-1,-3], axis = 1)[:,None])
		vec2[time_range,:] = evects[inv,:-1,-2]/(np.linalg.norm(evects[inv,:-1,-2], axis = 1)[:,None])
		vec3[time_range,:] = evects[inv,:-1,-1]/(np.linalg.norm(evects[inv,:-1,-1], axis = 1)[:,None])
		kdist[time_range] = distances[inv,-1]
	return val1, val2, val3, vec1, vec2, vec3, k, kdist