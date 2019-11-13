from laspy.file import File
from laspy.header import Header
from DEMO_pack.parallelism import splitter
from DEMO_pack import lasmaster
lm = lasmaster
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from time import time

@splitter
def add_pid(tile):
	inFile = File(tile)
	outFile = File("ID_"+tile, mode = "w", header = inFile.header)
	dimensions = [spec.name for spec in inFile.point_format]
	if not("slpid" in dimensions):
			outFile.define_new_dimension(name = "slpid", data_type = 7, description = 'Semantic Landscapes PID')
	outFile.writer.set_dimension("slpid", np.arange(len(inFile), dtype = np.uint64))	
	for dim in dimensions:
		if dim != "slpid":
			dat = inFile.reader.get_dimension(dim)
			outFile.writer.set_dimension(dim, dat)
	outFile.close()

def add_hag(tiles, vox, alpha):
	@splitter
	def f(tile):
		cf = {
			"vox"	:	vox,
			"alpha"	:	alpha,
			}
		lm.lpinteraction.add_hag(tile, cf)
	f(tiles)

def classify_ground(tiles):
	@splitter
	def run_pdal_ground(tile):
	#	ground_command = "pdal ground --initial_distance 1.0 --writers.las.extra_dims=all -i {} -o {}"
		ground_command = "pdal translate " \
		"--readers.las.extra_dims=\"slpid=uint64\" "\
		"--writers.las.extra_dims=all {} {} smrf" \
	#	" --filters.smrf.slope={} " \
	#	"--filters.smrf.cut={} " \
	#	"--filters.smrf.window={} " \
	#	"--filters.smrf.cell={} " \
	#	"--filters.smrf.scalar={} " \
		"--filters.smrf.threshold=1.0"
		command = ground_command.format(tile, "ground_"+tile)
		#command = ground_command.format(tile, "ground_"+tile)
		os.system(command)
		inFile = File("ground_"+tile, mode = "rw")
		ground = inFile.classification == 2
		classn = 1*inFile.classification
		classn[ground]=6
		classn[~ground]=0
		inFile.classification = classn
		inFile.close()
	run_pdal_ground(tiles)

def ground_removal(tiles):
	@splitter
	def split_off_grd(tile):
		inFile = File(tile, mode = "rw")
		points = inFile.points
		outFile1 = File("notgrd_"+tile, mode = "w", header = inFile.header)
		outFile2 = File("grd_"+tile, mode = "w", header = inFile.header)
		ground = inFile.classification == 6
		outFile2.points = points[ground]
		outFile2.close()
		outFile1.points = points[~ ground]
		outFile1.close()
	split_off_grd(tiles)

@splitter
def add_attributes(tile):
		cf = {
		"timeIntervals"	:	10,
		"k"				:	range(4,50), # must be a generator
		"radius"		:	0.5,
		"virtualSpeed"	:	0,
		"decimate"		:	0,
		}
		lm.lpinteraction.attr(tile, config = cf)

@splitter
def add_classification(tile):
	inFile = File(tile)

	def corridor(c, d, conductor_condition, R=1, S=2):
		if d == 4:
			v0 = 1*inFile.eig31[IND]
			v1 = 1*inFile.eig32[IND]
			v2 = 1*inFile.eig33[IND]
		if d == 3:
			v0 = 1*inFile.eig20[IND]
			v1 = 1*inFile.eig21[IND]
			v2 = 1*inFile.eig22[IND]
		cRestrict =  c[:, conductor_condition]
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(cRestrict))
		distances, indices = nhbrs.kneighbors(np.transpose(c))
		nns = indices[:,0]
		v = np.vstack((v0,v1,v2))[:, conductor_condition][:,nns] #(3,N)
		u = c[:,:]-c[:, conductor_condition][:,nns]
		scale =(u[0,:]*v[0,:]+u[1,:]*v[1,:]+u[2,:]*v[2,:])
		w = u-scale*v
		w_norms = np.sqrt(w[0,:]**2+w[1,:]**2+w[2,:]**2)
		condition = (w_norms<R)&(np.absolute(scale)<S)
		
		return condition

	voxel = inFile.vox
	UNQ, IND, INV, CNT = np.unique(voxel, return_index=True, return_inverse=True, return_counts=True)
	if (inFile.vox == 0).all():
		IND = np.arange(len(inFile))
		INV = IND
	u = 0.0*IND+0.05
	x = inFile.x[IND]
	y = inFile.y[IND]
	z = inFile.z[IND]
	dim1 = inFile.dim1[IND]
	dim2 = inFile.dim2[IND]
	dim3 = inFile.dim3[IND]
	eig0 = inFile.eig0[IND]
	eig1 = inFile.eig1[IND]
	eig2 = inFile.eig2[IND]

	classification = inFile.classification[IND]
	Coords = u[None,:]*np.floor(np.stack((x/u,y/u,z/u), axis = 0))
	LPS = np.stack((dim1, dim2, dim3), axis = 1)
	dims = 1+np.argmax(LPS, axis = 1)
	classn = 0*classification
	try:
		dim4 = inFile.dim4[IND]
		d = 4
	except:
		d = 3
	
	if d==4:
		eig3 = inFile.eig3

	if d == 3:
		dims[eig2<=0]=7
	if d == 4:
		dims[eig3<=0]=7

	if (dims == 1).any():
		if d == 4:
			v0 = 1*inFile.eig31[IND]
			v1 = 1*inFile.eig32[IND]
			v2 = 1*inFile.eig33[IND]
		if d == 3:
			v0 = 1*inFile.eig20[IND]
			v1 = 1*inFile.eig21[IND]
			v2 = 1*inFile.eig22[IND]
		condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
		v0[condition]=-v0[condition]
		v1[condition]=-v1[condition]
		v2[condition]=-v2[condition]
		v = np.vstack((5*v0,5*v1,5*v2, x, y, z))
		clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, dims == 1]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
			'A': labels,
			'X': x[dims == 1],
			'Y': y[dims == 1],
			'Z': z[dims == 1]
			}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn1 = classn[dims == 1]
		classn1[:] = 1
		classn1[lengths<=2]=0
		classn[dims == 1]=classn1
		if (classn == 1).any():
			conductor = corridor(Coords, d, classn == 1, R=0.5, S=2)
			classn[conductor]=1
		classn[dims == 7] = 7
	
	prepylon = (dims == 2)&(classn != 1)
	if prepylon.any():
		if d == 4:
			v0 = 1*inFile.eig21[IND]
			v1 = 1*inFile.eig22[IND]
			v2 = 1*inFile.eig23[IND]
		if d == 3:
			v0 = 1*inFile.eig10[IND]
			v1 = 1*inFile.eig11[IND]
			v2 = 1*inFile.eig12[IND]
		condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
		v0[condition]=-v0[condition]
		v1[condition]=-v1[condition]
		v2[condition]=-v2[condition]
		v = np.vstack((5*v0,5*v1,5*v2, x, y, z))
		clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, prepylon]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
			'A': labels,
			'X': x[prepylon],
			'Y': y[prepylon],
			'Z': z[prepylon],
			}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn2 = classn[prepylon]
		classn2[:] = 2
		classn2[lengths<=2]=0
		classn[prepylon]=classn2
		if (classn == 2).any():
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 2]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords))
			classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) ]=2

	preveg = (dims == 3) & (classn != 2) & (classn != 1)
	if preveg.any():
		v = np.vstack((x, y, z))
		clustering = DBSCAN(eps=0.5, min_samples=1).fit((v[:, preveg]).transpose(1,0))
		labels = clustering.labels_
		frame =	{
			'A': labels,
			'X': x[preveg],
			'Y': y[preveg],
			'Z': z[preveg],
			}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = (np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2))[inv]
		lengths[labels==-1]=0
		classn3 = classn[preveg]
		classn3[:] = 3
		classn3[lengths<2]=0
		classn[preveg] = classn3
		if (classn == 3).any():
			nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, classn == 3]))
			distances, indices = nhbrs.kneighbors(np.transpose(Coords))
			classn[(distances[:,0]<0.5)& (classn != 7) & (classn != 1) & (classn != 2)]=3
		
	if ((classn != 0) & (classn != 7)).any():
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(Coords[:, (classn != 0) & (classn != 7)]))
		distances, indices = nhbrs.kneighbors(np.transpose(Coords[:, classn == 0]))
		classn0 = classn[classn == 0]
		classn0[(distances[:,0]<0.5)] = (classn[(classn != 0) & (classn != 7)])[indices[(distances[:,0]<0.5),0]]
		classn[(classn==0)] = classn0

	outFile = File("classified"+tile, mode = "w", header = inFile.header)
	outFile.points = inFile.points
	outFile.classification = classn[INV]
	outFile.close()

def zip(tile_pairs):
	@splitter
	def zipper(tile_pair):
		tile1 = tile_pair[0]
		tile2 = tile_pair[1]
		name = "zipped_"+tile1[:-4]+tile2[:-4]+".las"		
		inFile1 = File(tile1)
		inFile2 = File(tile2)
		outFile = File(name, mode = "w", header = inFile1.header)
		outFile.x = np.concatenate((inFile1.x,inFile2.x))
		outFile.y = np.concatenate((inFile1.y,inFile2.y))
		outFile.z = np.concatenate((inFile1.z,inFile2.z))
		outFile.gps_time = np.concatenate((inFile1.gps_time,inFile2.gps_time))
		outFile.classification = np.concatenate((inFile1.classification,inFile2.classification))
		outFile.intensity = np.concatenate((inFile1.intensity,inFile2.intensity))
		outFile.return_num = np.concatenate((inFile1.return_num,inFile2.return_num))
		outFile.num_returns = np.concatenate((inFile1.num_returns,inFile2.num_returns))
		mask = (np.concatenate((np.ones(len(inFile1), dtype = int), np.zeros(len(inFile2), dtype = int)))).astype(bool)
		specs1 = [spec.name for spec in inFile1.point_format]
		specs2 = [spec.name for spec in inFile2.point_format]
		for dim in specs1:
			dat1 = inFile1.reader.get_dimension(dim)
			DAT = np.zeros(len(inFile1)+len(inFile2), dtype = dat1.dtype)
			DAT[mask] = dat1
			if dim in specs2:
				dat2 = inFile2.reader.get_dimension(dim)
				DAT[~mask] = dat2
			outFile.writer.set_dimension(dim, DAT)
		outFile.close()
	zipper(tile_pairs)

@splitter
def bbox(tile):
	inFile = File(tile)
	hag = inFile.hag
	theta = np.linspace(0, 2*np.pi, num = 1000)
	S = np.sin(theta)
	C = np.cos(theta)
	R = np.zeros((theta.size, 2, 2))
	R[:, 0, 0] = np.cos(theta)
	R[:, 0, 1] = -np.sin(theta)
	R[:, 1, 0] = np.sin(theta)
	R[:, 1, 1] = np.cos(theta)

	def bb(x, y, z, predicate):
		Coords = np.stack((x,y), axis = 0)
		coords_R = np.matmul(R, Coords[:, predicate])
		x_max = np.amax(coords_R[:, 0, :], axis = -1)
		y_max = np.amax(coords_R[:, 1, :], axis = -1)
		x_min = np.amin(coords_R[:, 0, :], axis = -1)
		y_min = np.amin(coords_R[:, 1, :], axis = -1)
		A = (x_max-x_min)*(y_max-y_min)
		k = np.argmin(A)
		R_min = R[k, : , :]
		Coords_R = np.matmul(R[k, :, :], Coords)
		predicate_bb = (x_min[k]-0.25<=Coords_R[0]) & (y_min[k]-0.25<=Coords_R[1]) & (x_max[k]>=Coords_R[0]-0.25) & (y_max[k]>=Coords_R[1]-0.25) & (max(z[predicate])+0.5>=z) & (min(z[predicate])-0.5<=z)
		return predicate_bb, A[k], min(x[predicate_bb]), max(x[predicate_bb]), min(y[predicate_bb]), max(y[predicate_bb])
		
	x = inFile.x
	y = inFile.y
	z = inFile.z
	coords = np.stack((x,y), axis = 1)
	out = File("bb_"+tile, mode = "w", header = inFile.header)
	out.points = inFile.points
	classn = np.ones(len(inFile), dtype = int)
	classn[:] = inFile.classification[:]

	classn_2_save = classn == 2
	if (classn==2).any():
		clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x,y,z), axis = 1)[classn_2_save, :])
		labels = clustering.labels_
		L = np.unique(labels)
		bldgs = np.empty((L.size, 6))
		i=0
		for item in L:
			predicate = np.zeros(len(inFile), dtype = bool)
			predicate[classn_2_save] = labels == item
			predicate_bb, area, x_min, x_max, y_min, y_max = bb(x, y, z, predicate)
			classn[predicate_bb] = 2
			bldgs[i] = [i, area, x_min, x_max, y_min, y_max]
			i+=1
			np.savetxt("buildings_"+tile[-4:]+".csv", bldgs, delimiter=",", header = "ID, Area, X_min, X_max, Y_min, Y_max")

	if (classn == 0).any() and (classn == 1).any():
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.stack((x,y), axis = 1)[classn == 1,:])
		distances, indices = nhbrs.kneighbors(np.stack((x,y), axis = 1)[classn == 0,:])
		classn0 = classn[classn == 0]
	
		try:
			angle = inFile.ang3
		except:
			angle = inFile.ang2


		
		classn0[(distances[:,0]< 1) & (angle[classn == 0] < 0.2)] = 5
		classn0[(distances[:,0]< 1) & (angle[classn == 0] < 0.2)] = 5
		classn[classn == 0] = classn0

		if (classn == 5).any():
			classn_5_save = classn == 5
			classn5 = classn[classn_5_save]
			unq, ind, inv = np.unique(np.floor(np.stack((x,y), axis = 1)[classn_5_save,:]).astype(int), return_index=True, return_inverse=True, return_counts=False, axis = 0)
			for item in np.unique(inv):
				z_max = np.max(z[classn_5_save][inv == item])
				z_min = np.min(z[classn_5_save][inv == item])
				if (z_max-z_min<5):
					classn5[inv==item]=0
			classn[classn_5_save]=classn5

		
		if (classn==5).any():
					classn_5_save = classn == 5
					clustering = DBSCAN(eps=0.5, min_samples=1).fit(np.stack((x,y), axis = 1)[classn_5_save, :])
					labels = clustering.labels_
					L = np.unique(labels)
					
					for item in L:
						predicate = np.zeros(len(inFile), dtype = bool)[(classn==0)|classn_5_save]
						predicate[classn_5_save[(classn==0)|classn_5_save]] = labels == item
						predicate_bb, area, x_min, x_max, y_min, y_max = bb(x[(classn==0)|classn_5_save], y[(classn==0)|classn_5_save], z[(classn==0)|classn_5_save], predicate)
						classn05 = classn[(classn==0)|classn_5_save]
						classn05[predicate_bb] = 5
						classn[(classn==0)|classn_5_save]=classn05

	out.classification = classn
	out.close()

@splitter
def conductor_matters(tile):
	inFile = File(tile, mode = "r")
	x = inFile.x
	y = inFile.y
	z = inFile.z
	hag = inFile.hag
	classn = np.zeros(len(inFile))
	classn = 1*inFile.classification
	cond = classn == 1
	if cond.any():
		clustering = DBSCAN(eps=2.5, min_samples=1).fit(np.stack((x,y,z), axis = 1)[cond, :])
		labels = clustering.labels_
		frame =	{
			'A': labels,
			'X': x[cond],
			'Y': y[cond],
			'Z': z[cond],
			'H': hag[cond]
			}
		df = pd.DataFrame(frame)
		maxs = (df.groupby('A').max()).values
		mins = (df.groupby('A').min()).values
		lq = (df.groupby('A').quantile(0.5)).values
		unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
		lengths = np.sqrt((maxs[:,0]-mins[:,0])**2+(maxs[:,1]-mins[:,1])**2+(maxs[:,2]-mins[:,2])**2)[inv]
		hags = lq[inv,3]
		lengths[labels==-1]=0
		classn1 = classn[cond]
		classn1[:] = 1
		classn1[lengths<=4]=0
		classn1[hags<7] = 0
		classn[cond]=classn1

	if (classn == 1).any() and (classn == 3).any():
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.stack((x,y,z), axis = 1)[classn == 1, :])
		distances, indices = nhbrs.kneighbors(np.stack((x,y,z), axis = 1)[classn == 3, :])
		classn3 = classn[classn==3]
		classn3[distances[:,0]<3]=4
		classn[classn==3]=classn3

	outFile = File("cm_"+tile, mode = "w", header = inFile.header)
	outFile.points = inFile.points
	outFile.classification = classn
	outFile.close()

def finisher(tile_original_pairs):
	@splitter
	def finish_tile(pair):
		tile, original = pair[0], pair[1]
		ogFile = File(original)
		inFile = File(tile)
		hag = inFile.hag
		hd = ogFile.header
		if os.path.exists("RESULTS"):
			pass
		else:
			os.system("mkdir RESULTS")
		os.chdir("RESULTS")
		outFile = File(original, mode = "w", header = hd)

		args = np.argsort(inFile.slpid)

		classn = 1*inFile.classification
		classn0 = classn == 0
		classn1 = classn == 1
		classn2 = classn == 2
		classn3 = classn == 3
		classn4 = classn == 4
		classn5 = classn == 5
		classn6 = classn == 6
		classn[classn0]=0
		classn[classn1]=14
		classn[classn2]=6
		classn[classn3]=5
		classn[classn4]=5
		classn[classn5]=15
		classn[classn6]=2
		lo = (0.5 < hag) & (hag <= 2)
		med = (2 < hag) & (hag <= 5)
		hi = (5 < hag)
		veg = classn3
		classn[veg] = 0
		classn[lo & veg] = 3
		classn[med & veg] = 4
		classn[hi & veg] = 5
		classn[(classn == 15) & (hag < 2)] = 0
		classn[classn == 7] = 0
		#classn[(hag < 0.5) & (classn != 6)] = 2
		classn[(classn == 14) & (hag < 2)] = 0
		classn[(hag < 0.5) & (classn == 0)] = 2

		for spec in ogFile.point_format:
			outFile.writer.set_dimension(spec.name, inFile.reader.get_dimension(spec.name)[args])
		outFile.classification = classn[args]
		outFile.x = inFile.x[args]
		outFile.y = inFile.y[args]
		outFile.z = inFile.z[args]
		os.chdir("..")
	finish_tile(tile_original_pairs)