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