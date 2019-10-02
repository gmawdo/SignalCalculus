import numpy as np 
import matplotlib.pyplot as plt
from laspy.file import File
from laspy.header import Header
from sklearn.neighbors import NearestNeighbors
import time
import pandas as pd

start = time.time()

flying_height = 500
FOV = 39.0 # degrees, for full left-right angle
SR = 40.0 # scan rate, Hz, lines per second, LPS
pulse_rate = 450000
speed_kts = 110 # converted from 110 kts to m/s
x_range = 100 # meters
mode = "shm" # "shm" or "linear"
density_mode = "radial" # "none" or "voxel" or "radial"
scale = 1

speed_mps = 0.514444*speed_kts
flight_time = x_range/speed_mps
num_pts = int(np.ceil(flight_time*pulse_rate))
times = np.arange(num_pts)*(1/pulse_rate)
print("mode:", mode)
print("density mode:", density_mode)
print("num. of points", num_pts)
print("pulse rate", pulse_rate)
print("speed", speed_mps)
print("flight range", x_range)
print("time elapsed", max(times)-min(times))
print("scan rate", SR)
print("field of view", FOV)

# make the aircraft position
x_ = np.empty(num_pts, dtype = float)
y_ = np.empty(num_pts, dtype = float)
z_ = np.empty(num_pts, dtype = float)

x_[:] = 0 # assume aircraft flies parrallel to y axis
y_ = times * speed_mps
z_[:] = flying_height

# make the scan position
x = np.empty(num_pts, dtype = float)
y = np.empty(num_pts, dtype = float)
z = np.empty(num_pts, dtype = float)
intensity = np.empty(num_pts, dtype = int)

angular_amplitude = (FOV)*(np.pi/180) # formulas subject to change when more understanding gained
frequency = SR # formulas subject to change when more understanding gained

if mode == "shm":
	function = lambda x: np.cos(2*np.pi*x)
if mode == "linear":
	function = lambda x: (2*(2*x-np.floor(2*x))-1)*((-1)**((np.floor(2*x)).astype(int)))

theta = angular_amplitude*function(times*frequency)

x = z_ * np.tan(theta)
y[:] = y_[:]
z[:] = 0

coords = np.stack((x,y,z), axis = 1)
differences = coords[:-1, :]-coords[1:,:]
distances_across_track = np.sqrt(np.sum(differences**2, axis = 1))

print("max pt spacing across track", np.max(distances_across_track))
print("max pt spacing along track", speed_mps/frequency)
print("swath width", max(x)-min(x))

unq, ind, inv, cnt = np.unique((np.floor(coords)).astype(int), return_index=True, return_inverse=True, return_counts=True, axis=0)
print("voxel mode lowest ppm", min(cnt), "(always given, even when radial selected)")
print("voxel mode highest ppm", max(cnt), "(always given, even when radial selected)")
print("voxel mode avg ppm", np.mean(cnt[inv]), "(always given, even when radial selected)")
unq1, ind1, inv1, cnt1 = np.unique((np.floor(coords/0.1)).astype(int), return_index=True, return_inverse=True, return_counts=True, axis=0)
frame =	{
		'A': inv,
		'B': inv1
		}
df = pd.DataFrame(frame)
nunique_vals = ((df.groupby('A')['B'].nunique()).values)[inv]

print("lowest coverage", min(nunique_vals), "percent")
print("highest coverage", max(nunique_vals), "percent")
print("avg coverage", np.mean(nunique_vals), "percent")

if density_mode == "voxel":
	ppm = cnt[inv]

if density_mode == "none":
	ppm = np.zeros(num_pts)

if density_mode == "radial":
	radius = np.sqrt(scale/np.pi)
	done = np.zeros(num_pts, dtype = bool)
	ppm = np.empty(num_pts, dtype = int)
	k = 1
	while not(done.all()):
		nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(coords[:, :2])
		distances, indices = nhbrs.kneighbors(coords[~ done, :2]) # (num_pts,k)
		k_found = distances[:, -1] >= radius
		not_done_save = ~done
		done[~done] = k_found
		ppm_not_done = ppm[not_done_save]
		ppm_not_done[k_found] = k
		ppm[not_done_save] = ppm_not_done
		k += 1
	print("radial mode lowest ppm", min(ppm))
	print("radial mode highest ppm", max(ppm))
	print("radial mode avg ppm", np.mean(ppm))

newHeader = Header()
newFile = File("als_model.las", mode = "w", header = newHeader)
newFile.header.scale = [0.0001, 0.0001, 0.0001] #4 dp
newFile.x = x
newFile.y = y
newFile.z = z
newFile.intensity = ppm
newFile.close()


#num_x = int((max(x)-min(x))/scale)
#num_y = int((max(y)-min(y))/scale)
#x_grid = scale*(int(min(x)/scale)+np.arange(num_x))
#y_grid = scale*(int(min(y)/scale)+np.arange(num_y))
#mesh = np.meshgrid(x_grid, y_grid)

#coords_grid = np.stack((mesh[0].flatten(), mesh[1].flatten(), np.zeros(mesh[0].size, dtype = float)), axis = 1)
#tree = BallTree(coords)
#radius = np.sqrt((scale/np.pi))
#balls = tree.query_radius(coords_grid, r=radius)

#grid = File("grid_als_model.las", mode = "w", header = newHeader)
#grid.header.scale = [0.0001, 0.0001, 0.0001]
#grid.x = coords_grid[:,0]
#grid.y = coords_grid[:,1]
#grid.z = coords_grid[:,2]
#grid_intensity = np.zeros(coords_grid.shape[-2], dtype = int)
#for item in range(coords_grid.shape[-2]):
#	grid_intensity[item] = len(balls[item])
#grid.intensity = grid_intensity

#grid.close()

end = time.time()
print((end-start)/60, "minutes")