import numpy as np 
import matplotlib.pyplot as plt
from laspy.file import File
from laspy.header import Header
from sklearn.neighbors import BallTree

flying_height = 500
FOV = 39.0 # degrees, for full left-right angle
SR = 40.0 # scan rate, Hz, lines per second
pulse_rate = 450000
speed_kts = 110 # converted from 110 kts to m/s
x_range = 100 # meters

speed_mps = 0.514444*speed_kts
flight_time = x_range/speed_mps
num_pts = int(np.ceil(flight_time*pulse_rate))
times = np.arange(num_pts)*(1/pulse_rate)
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

angular_amplitude = FOV*np.pi/360 # FOV needs to be halved, this is why we divide by 360 rather than 180
angular_frequency = 2*SR
theta = angular_amplitude*np.cos(2*np.pi*times*angular_frequency)

x = z_ * np.tan(theta)
y[:] = y_[:]
z[:] = 0

coords = np.stack((x,y,z), axis = 1)
differences = coords[:-1, :]-coords[1:,:]
distances_across_track = np.sqrt(np.sum(differences**2, axis = 1))

print("max pt spacing across track", np.max(distances_across_track))
print("max pt spacing along track", speed_mps/angular_frequency)
print("swath width", max(x)-min(x))

newHeader = Header()

newFile = File("als_model.las", mode = "w", header = newHeader)
newFile.header.scale = [0.0001, 0.0001, 0.0001]
newFile.x = x
newFile.y = y
newFile.z = z
newFile.close()

scale = 1

num_x = int((max(x)-min(x))/scale)
num_y = int((max(y)-min(y))/scale)
x_grid = scale*(int(min(x)/scale)+np.arange(num_x))
y_grid = scale*(int(min(y)/scale)+np.arange(num_y))
mesh = np.meshgrid(x_grid, y_grid)

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
