# DO NOT RE-ORDER THESE IMPORTS

from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import time

wpd_config = 	{
			"timeIntervals"	:	6,
			"k"		:	50,
			"radius"	:	0.5,
			"virtualSpeed"	:	2,
			"decimation"	:	0.1,
		}

histo_config =	{}


def wpd_attr(file_name):
	start = time.time()
	lpinteraction.attr(file_name, wpd_config, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def wpd_histo(file_name, attr = ["ent", "iso"]):
	in_file = lpinteraction.File(file_name, mode = "r")
	header = in_file.header
	
	coord_dictionary = {"x": in_file.x, "y": in_file.y, "z": in_file.z, "gps_time": in_file.gps_time}
	attr_dictionary = {name: in_file.reader.get_dimension(name) for name in attr}
	
	return geo.histogram(coord_dictionary, attr_dictionary, wpd_config, num_bins = 50)

def correlator(in_file, attributes, example, name, N=20, L=3, plot = False):
	from laspy.file import File
	import numpy as np

	# E N T R O P Y   O F   P R O B A B I L I T Y   V E C T O R S
	# input: (...,N) probability vectors
	# that is, we must have np.sum(distribution, axis = -1)=1
	# output: entropies of vectors
	def entropy(distribution):
		N = distribution.shape[-1]
		logs = np.log(distribution)
		logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
		entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
		return entropies

	# J E N S E N - S H A N N O N   D I V E R G E N C E
	# input: (...,M,N) matrices whose rows are probability vectors
	# output: J-S div. of the collection of vectors
	def jsd(distribution):
		M = distribution.shape[-2]
		N = distribution.shape[-1]
		return (entropy(np.mean(distribution, axis = -2))-np.mean(entropy(distribution), axis = -1))*np.log(N)/np.log(M)


	unq, ind, inv, cnt = np.unique(np.floor(np.vstack((in_file.x,in_file.y,in_file.z))/L), return_index = True, return_inverse = True, return_counts = True, axis=1)
	intensity = 0*in_file.intensity

	# get attributes from example data set
	example_attributes = {}
	for dimension in attributes:
		example_attributes[dimension]=example.reader.get_dimension(dimension)	

	# make the histogram "wavelet" from the data set
	example_digits = {}
	for dimension in attributes:
		example_digits[dimension] = np.digitize(example_attributes[dimension], bins = np.linspace(0,1,N+1))-1

	example_ids = sum(((N**index)*example_digits[dimension] for index, dimension in enumerate(attributes)))
	example_histogram, example_edges = np.histogram(example_ids, bins = range(N**len(attributes)), density = True)
	
	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

	# find JSD from "wavelet" to patches of "signal"
	m = len(np.unique(inv))
	for entry in np.unique(inv):
		voxel_attributes = {}
		for dimension in attributes:
			voxel_attributes[dimension]=in_file.reader.get_dimension(dimension)[inv == entry]
			
		
		voxel_digits = {}
		for dimension in attributes:
			voxel_digits[dimension] = np.minimum(np.digitize(voxel_attributes[dimension], bins = np.linspace(0,1,N+1)),N)-1
		
		voxel_ids = sum(((N**index)*voxel_digits[dimension] for index, dimension in enumerate(attributes)))
		voxel_histogram, voxel_edges = np.histogram(voxel_ids, bins = range(N**len(attributes)), density = True)
		intensity[inv == entry]=1000*(1-jsd(np.stack((voxel_histogram, example_histogram))))
		if plot:
			xs = voxel_edges[:-1]
			ys = voxel_histogram
			zs = entry
			ax.bar(xs, ys, zs, zdir='y', alpha=0.8)
		print(entry*100/m)


	outFile = File(name, mode = "w", header = in_file.header)
	outFile.points = in_file.points
	outFile.intensity = intensity

	if plot:
		ax.set_xlabel('Attribute_Digits')
		ax.set_ylabel('Voxel')
		ax.set_zlabel('Probability')
		plt.show()
	outFile.close()

