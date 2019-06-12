import os
from laspy.file import File
import numpy as np
from sklearn.neighbors import NearestNeighbors

for file in os.listdir():
	if "Completa" in file and "attr" in file and file[0]=="a":
		inFile = File(file)
		outFile = File("Conductor"+file, mode = "w", header = inFile.header)
		outFile.points = inFile.points
		Classification = inFile.classification
		Classification[:] = 0

		Stack = np.stack((inFile.linearity, inFile.planarity, inFile.scattering), axis = 1)
		Dim = np.argmax(Stack, axis=1)
		coords = np.vstack((inFile.x, inFile.y, inFile.z))
		nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(coords[:, Dim>0]))
		distances, indices = nhbrs.kneighbors(np.transpose(coords[:, Dim==0]))
		Class_zero = Classification[Dim==0]
		Class_zero[:] = 0
		Class_zero[distances[:,0]>=1] = 1
		Classification[Dim==0] = Class_zero

		import lasmaster as lm
		config = 	{
						"timeIntervals"	:	6,
						"k"		:	50,
						"radius"	:	1.0,
						"virtualSpeed"	:	2,
						"decimation"	:	0.0,
						}
		
		J = 0
		while J<10:

			coord_dictionary = {"x": inFile.x[Classification == 1], "y": inFile.y[Classification == 1], "z": inFile.z[Classification == 1], "gps_time": inFile.gps_time[Classification == 1]}
			val1, val2, val3, vec1, vec2, vec3, k, kdist = lm.geo.geo(coord_dictionary, config)

			linearity = lm.fun.std_fun_eig()["linearity"](val1, val2, val3)
			planarity = lm.fun.std_fun_eig()["planarity"](val1, val2, val3)
			scattering = lm.fun.std_fun_eig()["scattering"](val1, val2, val3)
			linearity[np.isnan(linearity)|np.isinf(linearity)] = 0
			planarity[np.isnan(planarity)|np.isinf(planarity)] = 0
			scattering[np.isnan(scattering)|np.isinf(scattering)] = 0

			ISO = lm.fun.std_fun_eig()["iso"](val1, val2, val3)
			ISO[np.isnan(ISO)|np.isinf(ISO)] = 0

			Stack = np.stack((linearity, planarity, scattering), axis = 1)
			protoclass = np.argmax(Stack, axis=1)

			Classification[Classification == 1] = 1*((protoclass == 0))
			J = J+1
		#END LOOP


		outFile.classification = Classification
		outFile.close()
