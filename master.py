# master.py -- A MODULE OF ALL FUNCTIONS USED

# This module requires numpy and laspy to be installed
import numpy
import laspy

# RETURNS THE ABSOLUTE VALUE OF LOCAL COVARIANCE ABOUT TWO AXES
# 0 is x, 1 is y, 2 is z
# returns nan if chosen sphere is empty
def labscov(num_1,num_2,in_file,a,b,c,radius): 
	indices_for_calc = (in_file.x-a)**2 + (in_file.y-b)**2 + (in_file.z-c)**2 < radius**2
	if not(numpy.isin(True, indices_for_calc)):
		return numpy.nan
	else: 
		x_for_calc = in_file.x[indices_for_calc]
		y_for_calc = in_file.y[indices_for_calc]
		z_for_calc = in_file.z[indices_for_calc]
		coords_for_calc = numpy.vstack((x_for_calc,y_for_calc,z_for_calc))
		cov_mat = numpy.cov(coords_for_calc)
		if len(x_for_calc) == 0:
			return 0
		else:
			return abs(cov_mat[num_1,num_2])

# FINDS OUT WHETHER A LOCAL NBHD OF A POINT CONTAINS A CHOSEN NUMBER OF PTS
#Returns true if (x,y,z) has more than N pts within a sphere of given radius
def nhbrs(N,in_file,a,b,c,radius):
	indices_for_calc = (in_file.x-a)**2 + (in_file.y-b)**2 + (in_file.z-c)**2 < radius**2
	pts = in_file.points
	pts_in_sphere = pts[indices_for_calc]
	if len(pts_in_sphere)>N:
		return True
	else:
		return False

# LOCAL STANDARD DEVIATION
# computes local standard deviation along a chosen axis
def lstd(num,in_file, a, b, c, radius):
	return numpy.sqrt(labscov(num,num,in_file,a,b,c,radius))

# COMPUTES LOCAL ABSOLUTE REGRATION (close to 1 if in a region of correlation, else close to 0)
def lreg(in_file, a, b, c, radius):
	axes = []
	for axis in range(2):
		if lstd(axis,in_file,a,b,c,radius) > 0:
			axes = axes + [axis]
	if len(axes) == 0:
		return numpy.nan
	if len(axes) == 1:
		return 1
	if len(axes) == 2:
		return labscov(axes[0],axes[1],in_file,a,b,c,radius)/(lstd(axes[0],in_file,a,b,c,radius)*lstd(axes[1],in_file,a,b,c,radius))
	if len(axes) == 3:
		return labscov(0,1,in_file,a,b,c,radius)*labscov(1,2,in_file,a,b,c,radius)*labscov(2,1,in_file,a,b,c,radius)/(lstd(0,in_file,a,b,c,radius)*lstd(1,in_file,a,b,c,radius)*lstd(2,in_file,a,b,c,radius))


# THROW AWAY FLIGHT LINE AND KEEP ONE OF MANY RETURNS
# in_file is a LAS file loaded into laspy
# returns a file in read mode where flight line is taken away and only one of manys kept
# produces a las file along the way, with the chosen name 
# name should not have .las extension
def one_of_many(in_file, name):
	off_flight  = in_file.classification != 10
	out_put_las = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	one_of_manys = numpy.logical_and(in_file.return_num == 1, in_file.num_returns!=1)
	indices_kept = numpy.logical_and(off_flight,one_of_manys)
	out_put_las.points = in_file.points[indices_kept]
	out_put_las.close()
	out_put_to_be_read = laspy.file.File(name+".las", mode = "r")
	return out_put_to_be_read

#I NPUT A LAS FILE AND COLOUR PTS IN REGION OF CORRELATION 
# in_file is the las file to be coloured as class 4
# the function simply creates a las file with those points coloured
# radius relates to the size of spheres used for local correlation
# name should be a string *without* .las extension
def crltd_pts(in_file, radius, name, corr_clip = 0.95):
	out_file = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	points = in_file.points
	xs = in_file.x
	ys = in_file.y
	zs = in_file.z
	out_file.points = points
	N = len(points)
	for marker in range(N):
		condition_01 = abs(labscov(0,1,in_file,xs[marker],ys[marker],zs[marker],radius))>=corr_clip*lstd(0,in_file,xs[marker],ys[marker],zs[marker],radius)*lstd(1,in_file,xs[marker],ys[marker],zs[marker],radius)
		condition_12 = abs(labscov(1,2,in_file,xs[marker],ys[marker],zs[marker],radius))>=corr_clip*lstd(1,in_file,xs[marker],ys[marker],zs[marker],radius)*lstd(2,in_file,xs[marker],ys[marker],zs[marker],radius)
		condition_20 = abs(labscov(2,0,in_file,xs[marker],ys[marker],zs[marker],radius))>=corr_clip*lstd(2,in_file,xs[marker],ys[marker],zs[marker],radius)*lstd(0,in_file,xs[marker],ys[marker],zs[marker],radius)
		if condition_01 and condition_12 and condition_20 and nhbrs(3,in_file,xs[marker],ys[marker],zs[marker],radius): #disregards pts with fewer than 3 neighbours, as
			out_file.raw_classification[marker]=4
		else:
			pass
	out_file.close()

def three_mesh(L_1,L_2,L_3):
	for i in L_1:
		for j in L_2:
			for k in L_3:
				yield [i,j,k]

#OUTPUTS LOCAL CORRELATION PLOT FROM A LAS FILE PLOTTED AS A LAS FILE
# in_file is the las file to be taken as our input data set
# the function creates a las file which forms a lattice, with the local correlation at each lattice point shown
# the function also takes in a minimum detail, which is the resolution of our grid
# name should be a string *without* .las extension
def lcorrplot(in_file, radius, name):
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	N_x = int((max_x-min_x)/radius)+1
	N_y = int((max_y-min_y)/radius)+1
	N_z = int((max_z-min_z)/radius)+1
	N = N_x*N_y*N_z
	h_x = (max_x-min_x)/N_x
	h_y = (max_y-min_y)/N_y
	h_z = (max_z-min_z)/N_z
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	for dex in lattice:
		xs.append(min_x+h_x*(dex[0]+0.5))
		ys.append(min_y+h_x*(dex[1]+0.5))
		zs.append(min_z+h_x*(dex[2]+0.5))
	out_file = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	out_file.x = numpy.array(xs)
	out_file.y = numpy.array(ys)
	out_file.z = numpy.array(zs)
	out_file.return_num = numpy.array([1]*N)
	out_file.num_returns = numpy.array([2]*N)
	out_file.raw_classification = numpy.array([1]*N)
	for index in range(N):
		correlation = lreg(in_file, out_file.x[index], out_file.y[index], out_file.z[index], radius)
		if numpy.isnan(correlation):
			out_file.intensity[index]=0
		else:
			out_file.intensity[index]=1000*correlation
	out_file.close()

# LOCAL STANDARD DEVIATION OF INTENSITY
# computes local standard deviation of intensity 
def lstdint(in_file, a, b, c, radius):
	indices_for_calc = (in_file.x-a)**2 + (in_file.y-b)**2 + (in_file.z-c)**2 < radius**2
	if not(numpy.isin(True, indices_for_calc)):
		return numpy.nan
	else: 
		intsty_for_calc = in_file.intensity[indices_for_calc]
		return numpy.std(intsty_for_calc)

# OUTPUTS LOCAL STANDARD DEVIATION OF INTENSITY PLOT FROM A LAS FILE PLOTTED AS A LAS FILE
# in_file is the las file to be taken as our input data set
# the function creates a las file which forms a lattice, with the local correlation at each lattice point shown
# the function also takes in a minimum detail, which is the resolution of our grid
# name should be a string *without* .las extension
# nans are set to zero -- afterall, our signals are meant to be detecting something! 
def lstdintplot(in_file, radius, name):
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	N_x = int((max_x-min_x)/radius)+1
	N_y = int((max_y-min_y)/radius)+1
	N_z = int((max_z-min_z)/radius)+1
	N = N_x*N_y*N_z
	h_x = (max_x-min_x)/N_x
	h_y = (max_y-min_y)/N_y
	h_z = (max_z-min_z)/N_z
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	for dex in lattice:
		xs.append(min_x+h_x*(dex[0]+0.5))
		ys.append(min_y+h_x*(dex[1]+0.5))
		zs.append(min_z+h_x*(dex[2]+0.5))
	out_file = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	out_file.x = numpy.array(xs)
	out_file.y = numpy.array(ys)
	out_file.z = numpy.array(zs)
	out_file.return_num = numpy.array([1]*N)
	out_file.num_returns = numpy.array([2]*N)
	out_file.raw_classification = numpy.array([1]*N)
	for index in range(N):
		newintensity = lstdint(in_file, out_file.x[index], out_file.y[index], out_file.z[index], radius)
		if numpy.isnan(newintensity):
			out_file.intensity[index]=0
		else:
			out_file.intensity[index]=newintensity
	out_file.close()

# LOCAL COVARIANCE
# not absolute value
def lcov(num_1,num_2,in_file,a,b,c,radius): 
	indices_for_calc = (in_file.x-a)**2 + (in_file.y-b)**2 + (in_file.z-c)**2 < radius**2
	if not(numpy.isin(True, indices_for_calc)):
		return numpy.nan
	else: 
		x_for_calc = in_file.x[indices_for_calc]
		y_for_calc = in_file.y[indices_for_calc]
		z_for_calc = in_file.z[indices_for_calc]
		coords_for_calc = numpy.vstack((x_for_calc,y_for_calc,z_for_calc))
		cov_mat = numpy.cov(coords_for_calc)
		if len(x_for_calc) == 0:
			return 0
		else:
			return cov_mat[num_1,num_2]
		
# LOCAL COPLANARITY
# computes local coplanarity/planar regression
def lcoplan(in_file, a, b, c, radius):
	exp1 = (lstd(0,in_file,a,b,c,radius)*lstd(1,in_file,a,b,c,radius)*lstd(2,in_file,a,b,c,radius))**2+2*lcov(0,1,in_file,a,b,c,radius)*lcov(1,2,in_file,a,b,c,radius)*lcov(2,0,in_file,a,b,c,radius)
	exp2 = (lstd(0,in_file,a,b,c,radius)*lcov(1,2,in_file,a,b,c,radius))**2+(lstd(1,in_file,a,b,c,radius)*lcov(2,0,in_file,a,b,c,radius))**2+(lstd(2,in_file,a,b,c,radius)*lcov(1,0,in_file,a,b,c,radius))**2
	if exp1 == 0:
		return 1
	else:
		return exp2/exp1

# OUTPUTS LOCAL COPLANARITY PLOT FROM A LAS FILE PLOTTED AS A LAS FILE
# in_file is the las file to be taken as our input data set
# the function creates a las file which forms a lattice, with the local correlation at each lattice point shown
# the function also takes in a minimum detail, which is the resolution of our grid
# name should be a string *without* .las extension
# nans are set to zero -- afterall, our signals are meant to be detecting something!

def lcoplanplot(in_file, radius, name):
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	N_x = int((max_x-min_x)/radius)+1
	N_y = int((max_y-min_y)/radius)+1
	N_z = int((max_z-min_z)/radius)+1
	N = N_x*N_y*N_z
	h_x = (max_x-min_x)/N_x
	h_y = (max_y-min_y)/N_y
	h_z = (max_z-min_z)/N_z
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	for dex in lattice:
		xs.append(min_x+h_x*(dex[0]+0.5))
		ys.append(min_y+h_x*(dex[1]+0.5))
		zs.append(min_z+h_x*(dex[2]+0.5))
	out_file = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	out_file.x = numpy.array(xs)
	out_file.y = numpy.array(ys)
	out_file.z = numpy.array(zs)
	out_file.return_num = numpy.array([1]*N)
	out_file.num_returns = numpy.array([2]*N)
	out_file.raw_classification = numpy.array([1]*N)
	for index in range(N):
		val = lcoplan(in_file, out_file.x[index], out_file.y[index], out_file.z[index], radius)
		if numpy.isnan(val):
			out_file.intensity[index]=0
		else:
			out_file.intensity[index]=1000*val
	out_file.close()

def distbetweencenters(in_file, radius, name):
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	max_x = max(x_array)
	max_y = max(y_array)
	max_z = max(z_array)
	min_x = min(x_array)
	min_y = min(y_array)
	min_z = min(z_array)
	N_x = int((max_x-min_x)/radius)+1
	N_y = int((max_y-min_y)/radius)+1
	N_z = int((max_z-min_z)/radius)+1
	N = N_x*N_y*N_z
	h_x = (max_x-min_x)/N_x
	h_y = (max_y-min_y)/N_y
	h_z = (max_z-min_z)/N_z
	lattice = three_mesh(range(N_x),range(N_y),range(N_z))
	xs = []
	ys = []
	zs = []
	for dex in lattice:
		xs.append(min_x+h_x*(dex[0]+0.5))
		ys.append(min_y+h_x*(dex[1]+0.5))
		zs.append(min_z+h_x*(dex[2]+0.5))
	out_file = laspy.file.File(name+".las", mode = "w", header = in_file.header)
	out_file.x = numpy.array(xs)
	out_file.y = numpy.array(ys)
	out_file.z = numpy.array(zs)
	out_file.return_num = numpy.array([1]*N)
	out_file.num_returns = numpy.array([2]*N)
	out_file.raw_classification = numpy.array([1]*N)
	for index in range(N):
		a, b, c = (out_file.x[index], out_file.y[index], out_file.z[index])
		indices_for_calc = (in_file.x-a)**2 + (in_file.y-b)**2 + (in_file.z-c)**2 < radius**2
		if True in indices_for_calc:
			xs = in_file.x[indices_for_calc]
			ys = in_file.y[indices_for_calc]
			zs = in_file.z[indices_for_calc]
			Intss = in_file.intensity[indices_for_calc]
			Is = Intss.astype(float)
			CoM = numpy.array([numpy.mean(xs),numpy.mean(ys).numpy.mean(zs)])
			CoI = numpy.array([numpy.mean(xs*Is)/numpy.mean(Is),numpy.mean(ys*Is)/numpy.mean(Is),numpy.mean(zs*Is)/numpy.mean(Is)])
			out_file.intensity[index]=1000*numpy.sqrt(numpy.sum((CoM-CoI)**2))/radius
		else:
			out_file.points[index]=0
	out_file.close()
