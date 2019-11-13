import numpy as np
np.seterr(divide='ignore', invalid='ignore')
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
	evals = evals/np.sum(evals, axis = -1)[..., None]
	return evals, evects # evects of shape (...,d,d) with evects[:,:,i] being vector i

# O P T I M I S E   K   N U M B E R S
def optimise_k(coords, distances, nbhds, k_range, condition = True, min_k = 4): # coords.shape == (...,n,d)
	# condition must be able to broadcast over coords to same shape as coords
	d = coords.shape[-1]
	k_opt = np.ones(coords.shape[:-1], dtype = int)
	entropy_store = np.ones(coords.shape[:-1], dtype = float)
	eval_store = np.empty(coords.shape[:-1] + (d,), dtype = float)
	evect_store = np.empty(coords.shape[:-1] + (d,d), dtype = float)
	unchanged = np.ones(coords.shape[:-1], dtype = bool)
	for item in (k for k in k_range if k >= min_k):
		evals, evects = eig_info(nbhds, coords, condition) #(num_pts,d), (num_pts,d,d)
		actions = np.zeros((d,d), dtype = float)
		for index in range(d):
			actions[index, d-index-1] = d-index
			actions[index, d-index-2] = index-d+1
		LPS = np.matmul(evals/(np.sum(evals, axis = -1)[...,None]), actions) #(..., d)
		dim_ent = np.clip(entropy(LPS), 0, 1)
		change_condition = dim_ent<=entropy_store
		k_opt[change_condition] = item
		entropy_store[change_condition] = dim_ent[change_condition]
		eval_store[change_condition,:] = evals[change_condition,:]
		evect_store[change_condition,:,:] = evects[change_condition,:,:]
		unchanged[change_condition] = False
	entropy_store[unchanged] = 0
	eval_store[unchanged,:] = 0
	evect_store[unchanged,:,:] = 0
	return k_opt, eval_store/(np.sum(eval_store, axis = -1)[...,None]), evect_store, entropy_store

def decimate(coords, u):
	if u == False:
		return coords, ..., ...
	else:
		markers = (np.floor(coords/u)).astype(int)
		unq, ind, inv, cnt = np.unique(markers, return_index=True, return_inverse=True, return_counts=True, axis=0)
		return u*unq, inv, ind

def attibutes_prelim(x,y,z,t, config):
	N = config["timeIntervals"]
	k_range = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimate"]
	spacetime = bool(v_speed)
	if spacetime:
		coords, inv, ind = decimate(np.stack((x,y,z,v_speed * t), axis = 1), u) 
	else:
		coords, inv, ind = decimate(np.stack((x,y,z), axis = 1), u) 
	time = t[ind]
	d = coords.shape[-1]
	times = [np.quantile(time, q = r) for r in np.linspace(0,1,N+1)]
	time_digits = np.digitize(time, bins = times, right = False) # bins 0,1,2,...,N+1. 0 represents less than min, N+1 more than max.
	time_digits[time_digits<1]=1
	time_digits[time_digits>N]=N
	val= np.empty(coords.shape, dtype = float)
	vec = np.empty(coords.shape+(d,), dtype = float)
	ents= np.empty(coords.shape[:-1], dtype = float)
	kdist = np.empty(coords.shape[:-1], dtype = float)
	kopt = np.empty(coords.shape[:-1], dtype = int)
	dist1 = np.empty(coords.shape[:-1], dtype = float)
	distmax = np.empty(coords.shape[:-1], dtype = float)
	try:
		nhbrs = NearestNeighbors(n_neighbors = max(k_range), algorithm = "kd_tree").fit(coords)
		for i in np.unique(time_digits):
			time_range = time_digits == i
			distances, indices = nhbrs.kneighbors(coords[time_range,:]) # (num_pts,k)
			condition = (distances[:,:,None] < radius)
			k_opt, evals, evects, entropies = optimise_k(coords[time_range,:], distances, coords[indices, :], k_range, condition) #(num_pts, d), (num_pts,d,d)
			val[time_range,:] = evals
			vec[time_range,:,:] = evects
			ents[time_range] = entropies
			kdist[time_range] = distances[np.arange(distances.shape[-2]), k_opt-1]
			kopt[time_range] = k_opt
			dist1[time_range] = distances[:,1]
			distmax[time_range] = distances[:,-1]
	except:
		print("Local geometry couldn't be done: points possibly not dense enough!")
	k_dictionary = {}
	kdist_dictionary = {}
	k_dictionary["one"] = np.ones(coords.shape[:-1], dtype = int)[inv]
	k_dictionary["max"] = max(k_range)*np.ones(coords.shape[:-1], dtype = int)[inv]
	k_dictionary["opt"] = kopt[inv]
	kdist_dictionary["one"] = dist1[inv]
	kdist_dictionary["max"] = distmax[inv]
	kdist_dictionary["opt"] = kdist[inv]
	return val[inv,:], vec[inv,:,:], k_dictionary, kdist_dictionary, inv

def hag(coord_dictionary, config):
	Coords = np.stack((coord_dictionary["x"],coord_dictionary["y"],coord_dictionary["z"]), axis = 1)
	alpha = config["alpha"] # alpha should be small
	voxel_size = config["vox"]
	unq, ind, inv, cnt = np.unique((np.floor(Coords[:,:2]/voxel_size)).astype(int), return_index=True, return_inverse=True, return_counts=True, axis=0)
	df = pd.DataFrame({'A': inv, 'Z': Coords[:, 2]})
	ground = (pd.DataFrame({'A':df['A'], 'Z':df['Z']}).groupby('A').quantile(alpha)).values
	df['Z'] = ground[df['A'],0]

	# new code below - can delete if don't like
	#unq_z = np.empty((unq.shape[-2],3), dtype = float)
	#unq_z[:, :2] = unq
	#unq_z[:, 2] = ground[:,0]
	#nhbrs = NearestNeighbors(n_neighbors = 3, algorithm = "kd_tree").fit(unq_z[:, :2])
	#distances, indices = nhbrs.kneighbors(unq_z[:, :2])
	#coords = unq_z[indices, :] # num_pts x (k=3) x 3 
	#xy1 = np.empty(coords.shape) # num_pts x (k=3) x 3
	#xy1[:] = coords[:]
	#z = coords[:, :, 2] # num_pts x (k=3)
	#xy1[:, :, 2] = 1.0
	#print(sum(1*(np.absolute(np.linalg.det(xy1)<0.00001))), coords.shape[-3])
	#abn = np.matmul(np.linalg.inv(xy1), z[:, :, None])[:, :, 0] # num_pts x 3
	#df['Z'] = abn[inv, 0]*Coords[0,:]+abn[inv, 1]*Coords[1,:]+abn[inv, 2]
	#end of new
	
	return coord_dictionary["z"]-df['Z']
