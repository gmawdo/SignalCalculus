def corridor(coords, eigenvectors, mask, R=1, S=2):
	"""

	Parameters:
	Find a cylindrical corridor around a family of points.

	This function takes a family of points, encoded in a numpy array coords; coords.shape == (n, d) where 
	n is the number of points and d is the number of dimensions.

	The eigenvectors give the eigenvectors which we want to corridor along.

	The parameter R dictates how wide our cylinder is.

	The parameter S dictates how long our cylinder is.

	The mask tells us which points in coords we are making a cylinder around.

	"""

	# apply the mask to the coordinates to get the points we are interested in drawing a corridor around
	coords_mask = coords[mask, :]
	# find nearest neighbours from each point to the points of interest
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(coords_mask)
	distances, indices = nhbrs.kneighbors(coords)
	nearest_nbrs = indices[:,0]

	# find direction from each point to point of interest, project onto eigenvector
	v = eigenvectors[nearest_nbrs]
	u = coords - coords[conductor_condition][nearest_nbrs]
	scale = np.sum(u*v, axis = 1)
	# find coprojection
	w = u - scale*v

	# find distance to line formed by eigenvector
	w_norms = np.sqrt(np.sum(w**2, axis = 1))

	# return condition for the corridor
	condition = (w_norms<R)&(np.absolute(scale)<S)
	return condition

def dimension(inFile):
	"""

	Parameters: 

	Requires only a file with dim1, dim2 and dim3 attributes.

	Outputs:

	Dimensions for each point

	"""
	# gather the three important attributes into an array
	LPS = np.stack((inFile.dim1, inFile.dim2, inFile.dim3))
	#compute dimensions 
	dims = 1 + np.argmax(LPS, axis= 1)
	return dims

def eigen_clustering(coords, eigenvector, tolerance, eigenvector_scale, max_length, min_pts):
	"""
	Parameters:
	points_to_cluster - a numpy index or mask.
	tolerance - how close do two points have to be in order to be in same cluster?
	max_length - how long can a cluster be?
	min_pts - how many points must a cluster have?

	Outputs:
	labels

	Extra notes:
	Eigenvectors should have unit length.
	"""
	x = coords[:, 0]
	y = coords[:, 1]
	z = coords[:, 2]
	v0 = eigenvector_scale*eigenvector[:, 0]
	v1 = eigenvector_scale*eigenvector[:, 1]
	v2 = eigenvector_scale*eigenvector[:, 2]
	condition = ((v0>=0)&(v1<0)&(v2<0))|((v1>=0)&(v2<0)&(v0<0))|((v2>=0)&(v0<0)&(v1<0))|((v0<0)&(v1<0)&(v2<0))
	v0[condition]=-v0[condition]
	v1[condition]=-v1[condition]
	v2[condition]=-v2[condition]
	clusterable = np.stack((x, y, z, u, v, w))
	clustering = DBSCAN(eps=tolerance, min_samples=min_pts).fit(clusterable)
	labels = clustering.labels_
	frame =	{
		'A': labels,
		'X': x,
		'Y': y,
		'Z': z,
		}
	df = pd.DataFrame(frame)
	maxs = (df.groupby('A').max()).values
	mins = (df.groupby('A').min()).values
	unq, ind, inv, cnt = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
	lengths = np.sqrt((maxs[inv,0]-mins[inv,0])**2+(maxs[inv,1]-mins[inv,1])**2)
	labels[lengths < max_length] = -1
	return labels

### now I will show how these can be combined to give the add_classification step
def add_classification(inFile):
	# load the voxel numbers - they will be only one (0) if no voxelisation happened
	voxel = inFile.vox
	# find how to map each point onto each voxel
	UNQ, IND, INV, CNT = np.unique(voxel, return_index=True, return_inverse=True, return_counts=True)
	# determine by number of voxel numbers whether decimation occured
	decimated = IND.size > 0
	# if no decimation occured this mapping must be trivial:
	if not decimated:
		IND = np.arange(len(inFile))
		INV = IND
	
	# grab the attributes we need - but only on decimated points
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

	# scale down coordinates
	if decimated:
		u = inFile.scale
		Coords = u[None,:]*np.floor(np.stack((x/u,y/u,z/u), axis = 1))
	else:
		Coords = np.stack((x,y,z), axis = 1)

	# build the probabilistic dimension
	LPS = np.stack((dim1, dim2, dim3), axis = 1)
	# extract a discrete dimension - 1, 2 or 3
	dims = 1+np.argmax(LPS, axis = 1)
	classn = np.zeros(classification.shape, dtype = classification.dtype)

	# here we decide whether 4 dimensions were run by trying to extract eigenvalue 3 - 3D space has eigenvalues 0, 1, 2  
	# so this throws an error if we are in 3D and runs if we are in 4D
	try:
		eig3 = inFile.eig3
		d = 4
	except:
		d = 3


	if d == 3:
		dims[eig2<=0]=0
	if d == 4:
		dims[eig3<=0]=0

	noise = dims == 0
	dim1 = dims == 1
	dim2 = dims == 2
	dim3 = dims == 3

	if (dims == 1).any():
		if d == 4:
			v0 = 1*inFile.eig30[IND]
			v1 = 1*inFile.eig31[IND]
			v2 = 1*inFile.eig32[IND]
		if d == 3:
			v0 = 1*inFile.eig20[IND]
			v1 = 1*inFile.eig21[IND]
			v2 = 1*inFile.eig22[IND]
	line_of_best_fit_direction = np.stack((v0, v1, v2), axis = 1)/np.sqrt(v0**2+v1**2+v2**2)
	labels = eigen_clustering(Coords, line_of_best_fit_direction, 0.5, 5.0, 2, 1)

	classn[dim1] = 1
	classn[labels = -1] = 0

	conductor = corridor(Coords, line_of_best_fit_direction[classn == 1], classn == 1, R=0.5, S=2)
	classn[conductor] = 1
	classn[noise] = 0

	mask = dim2 & (~ noise) & (classn != 1)
	if mask.any():
		if d == 4:
			v0 = 1*inFile.eig20[IND]
			v1 = 1*inFile.eig21[IND]
			v2 = 1*inFile.eig22[IND]
		if d == 3:
			v0 = 1*inFile.eig10[IND]
			v1 = 1*inFile.eig11[IND]
			v2 = 1*inFile.eig12[IND]
	plane_of_best_fit_direction = np.stack((v0, v1, v2), axis = 1)/np.sqrt(v0**2+v1**2+v2**2)