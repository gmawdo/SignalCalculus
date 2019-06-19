import numpy as np
from sklearn.neighbors import NearestNeighbors
from laspy.file import File

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

# matrix multiplications
raw_deviations = np.matmul(raw_deviations_prestack[:,None,:,:], J) # (d,k-aux+1,num_pts,k)
cov_matrices = np.matmul(raw_deviations.transpose(1,2,0,3), raw_deviations.transpose(1,2,3,0)) #(k-aux+1,num_pts,d,d)
cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,1,3,2)) #(k-aux+1,num_pts,d,d)
evals, evects = np.linalg.eigh(cov_matrices) #(k-aux+1,num_pts, d), (k-aux+1,num_pts,d,d)
print(evals.shape, evects.shape)
linearity = (evals[:,:,-1]-evals[:,:,-2])/evals[:,:,-1] #(k-aux+1,num_pts)
planarity = (evals[:,:,-2]-evals[:,:,-3])/evals[:,:,-1] #(k-aux+1,num_pts)
scattering = evals[:,:,-3]/evals[:,:,-1] #(k-aux+1,num_pts)