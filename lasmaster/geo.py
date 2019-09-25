import numpy as np
from sklearn.neighbors import NearestNeighbors
from lasmaster.infotheory import entropy
from lasmaster import fun
import pandas as pd

# C O M P U T E   E I G E N I N F O
def eig_info(coords, target, condition = True): # coords = coordinates, target = target point(s)
	# target.shape == (...,d) for single target, coords.shape == (...,k,d)
	deviations = condition*(coords - target[..., None, :]) # raw_deviations.shape == (...,k,d)
	cov_matrices = np.matmul(deviations.swapaxes(-1,-2), deviations) # (...,d,d)
	cov_matrices = np.maximum(cov_matrices.swapaxes(-1,-2), cov_matrices) # induce symmetry
	evals, evects = np.linalg.eigh(cov_matrices) #(...,d), (...,d,d)

	return evals, evects # evects of shape (...,d,d) with evects[:,:,i] being vector i

# O P T I M I S E   K   N U M B E R S
def optimise_k(coords, distances, indices, condition, min_k = 4, condition = True): # coords.shape == (...,n,d)
	# condition must be able to broadcast over coords to same shape as coords
	d = coords[-1]
	k_opt = np.ones(coords.shape[:-1], dtype = int)
	entropy_store = np.ones(coords.shape[:-1], dtype = float)
	eval_store = np.empty(coords.shape[:-1] + (d,))
	evect_store = np.empty(coords.shape[:-1] + (d,d))
	unchanged = np.ones(coords.shape[:-1], dtype = bool)

	for item in (k for k in k_range if k >= min_k):
		evals, evects = eig_info(coords[indices, :], coords, condition)
		actions = np.zeros((d,d), dtype = float)
		for index in range(d):
			actions[index, d-index-1] = d-index
			actions[index, d-index-2] = index-d+1
		LPS = np.matmum(evects, actions) #(..., d)
		dim_ent = np.clip(entropy(LPS), 0, 1)
		condition = dim_ent<=entropy_store
		k_opt[condition] = item
		entropy_store[condition] = dim_ent[condition]
		eval_store[condition,:] = evals[condition,:]
		evect_store[condition,:,:] = evects[condition,:,:]
		unchanged[condition] = False

	entropy_store[unchanged] = 0
	eval_store[unchanged,:] = 0
	evect_store[unchanged,:,:] = 0


	return k_opt, eval_store, evect_store

def eig(coord_dictionary, config):
	x = coord_dictionary["x"]
	y = coord_dictionary["y"]
	z = coord_dictionary["z"]
	t = coord_dictionary["gps_time"]

	N = config["timeIntervals"]
	k_range = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]

	spacetime = bool(v_speed)

	d = 3+int(spacetime)

	times = [np.quantile(t, q = r) for r in np.linspace(0,1,N+1)]
	time_digits = np.digitize(t, bins = times, right = False) # bins 0,1,2,...,N+1. 0 represents less than min, N+1 more than max.
	time_digits[time_digits<1]=1
	time_digits[time_digits>N]=N
	
	val1 = np.empty(t.shape, dtype = float)
	val2 = np.empty(t.shape, dtype = float)
	val3 = np.empty(t.shape, dtype = float)
	vec1 = np.empty(t.shape+(3,), dtype = float)
	vec2 = np.empty(t.shape+(3,), dtype = float)
	vec3 = np.empty(t.shape+(3,), dtype = float)
	kdist = np.empty(t.shape, dtype = float)
	kopt = np.empty(t.shape, dtype = int)
	dist1 = np.empty(t.shape, dtype = float)
	distmax = np.empty(t.shape, dtype = float)

	coords = np.vstack((x,y,z)+spacetime*(v_speed*t))
	nhbrs = NearestNeighbors(n_neighbors = max(k_range), algorithm = "kd_tree").fit(np.transpose(coords))

	for i in np.unique(time_digits):

		time_range = time_digits == i
		
		distances, indices = nhbrs.kneighbors(np.transpose(coords[:,time_range])) # (num_pts,k)
		relative_positions = (distances < radius)*(coords[:,indices] - coords[:,time_range][:,:,None]) #(d,num_pts,k)

		k_opt, evals, evects = optimise_k(relative_positions, k_range) #(num_pts, d), (num_pts,d,d)

		val1[time_range] = evals[:,-3]
		val2[time_range] = evals[:,-2]
		val3[time_range] = evals[:,-1]
		vec1[time_range,:] = evects[:,:,-3]/(np.linalg.norm(evects[:,:,-3], axis = 1)[:,None])
		vec2[time_range,:] = evects[:,:,-2]/(np.linalg.norm(evects[:,:,-2], axis = 1)[:,None])
		vec3[time_range,:] = evects[:,:,-1]/(np.linalg.norm(evects[:,:,-1], axis = 1)[:,None])
		kdist[time_range] = distances[np.arange(indices.shape[0]), k_opt-1]
		kopt[time_range] = k_opt
		dist1[time_range] = distances[:,1]
		distmax[time_range] = distances[:,-1]

	k_dictionary = {}
	kdist_dictionary = {}
	k_dictionary["one"] = np.ones(t.shape, dtype = int)
	k_dictionary["max"] = max(k_range)*np.ones(t.shape, dtype = int)
	k_dictionary["opt"] = kopt
	kdist_dictionary["one"] = dist1
	kdist_dictionary["max"] = distmax
	kdist_dictionary["opt"] = kdist
	return val1, val2, val3, vec1, vec2, vec3, k_dictionary, kdist_dictionary

def hag(coord_dictionary, config):
	Coords = np.vstack((coord_dictionary["x"],coord_dictionary["y"],coord_dictionary["z"]))
	alpha = config["alpha"] # alpha should be small
	voxel_size = config["vox"]
	unq, ind, inv, cnt = np.unique(np.round(Coords[:2,:]/voxel_size,0), return_index=True, return_inverse=True, return_counts=True, axis=1)
	df = pd.DataFrame({'A': inv, 'Z': Coords[2,:]})
	ground = (pd.DataFrame({'A':df['A'], 'Z':df['Z']}).groupby('A').quantile(0.01)).values
	df['Z'] = ground[df['A'],0]
	
	return coord_dictionary["z"]-df['Z']