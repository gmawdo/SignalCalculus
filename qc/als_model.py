import numpy as np 
import matplotlib.pyplot as plt

pulse_rate = 450000
speed = 56.5889 # converted from 110 kts to m/s
x_range = 100
flight_time = x_range/speed
num_pts = int(np.ceil(flight_time*pulse_rate))
print(num_pts, "points")
times = np.arange(num_pts)*(1/pulse_rate)
SR = 40.0 # scan rate, Hz, lines per second
FOV = 39.0*np.pi/180
print("pulse rate", pulse_rate)
print("speed", speed)
print("flight range", 100)
print("scan rate", SR)
print("field of view", SR)

# make the aircraft position
x_ = np.empty(num_pts, dtype = float)
y_ = np.empty(num_pts, dtype = float)
z_ = np.empty(num_pts, dtype = float)

x_[:] = 0 # assume aircraft flies parrallel to y axis
y_ = times * speed
z_[:] = 500

# make the scan position
x = np.empty(num_pts, dtype = float)
y = np.empty(num_pts, dtype = float)
z = np.empty(num_pts, dtype = float)

integer_floor = lambda w: w.astype(int)
zig_zag = lambda w: (2*(w-integer_floor(w))-1)*((-1)**integer_floor(w))

theta = zig_zag(times*SR)

x = z_ * np.tan(theta)
y[:] = y_[:]
z[:] = 0

print("swath width", max(x)-min(x))

np.savetxt('als_model.csv', np.stack((x,y,z), axis = 1))