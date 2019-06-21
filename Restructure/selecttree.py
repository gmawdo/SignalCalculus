import numpy as np
from sklearn.neighbors import NearestNeighbors
from laspy.file import File
import matplotlib.pyplot as plt

def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
	return entropies

def ladder_tensor(m, n):
	I = np.empty((n,n,n))
	for i in range(n):
		I[i,:,:]=np.identity(n)
		I[i,(i+1):,:]=0
	return I[m-1:,:,:] # returns an (n-m+1) x n x n, rank 3 tensor

inFile = File("T200_010_01_2017_04_20-05_03_22_2_090_TILE08_outputNFLClip100_00.las", mode = "rw")

Class = inFile.classification
Class[:] = 0


x = inFile.x
y = inFile.y
z = inFile.z

IDs = np.arange(len(inFile))

def round(vec):
	return (np.floor(100*vec)).astype(int)


x_intveg = 38531721
y_intveg = 20795766
z_intveg = 8249
ID = (IDs[(round(inFile.x) == x_intveg)&(round(inFile.y) == y_intveg)&(round(inFile.z) == z_intveg)])[0]
min_num = 2
N = 1
k = 100
d=3

Class[ID] = 1

inFile.classification = Class

coords = np.vstack((x, y, z))

aux = min_num

nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
distances, indices = nhbrs.kneighbors(np.transpose(coords[:,ID][:,None])) # (num_pts,k)
J = ladder_tensor(aux, k) # we use this tensor to avoid a loop # (k-aux+1, k, k)
raw_deviations_prestack = (coords[:,indices] - coords[:,ID][:,None,None]) # (d,num_pts,k)

# matrix multiplications
raw_deviations = np.matmul(raw_deviations_prestack[:,None,:,:], J) # (d,k-aux+1,num_pts,k)
cov_matrices = np.matmul(raw_deviations.transpose(1,2,0,3), raw_deviations.transpose(1,2,3,0)) #(k-aux+1,num_pts,d,d)
cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,1,3,2)) #(k-aux+1,num_pts,d,d)
evals, evects = np.linalg.eigh(cov_matrices) #(k-aux+1,num_pts, d), (k-aux+1,num_pts,d,d)
linearity = (evals[:,:,-1]-evals[:,:,-2])/evals[:,:,-1] #(k-aux+1,num_pts)
planarity = (evals[:,:,-2]-evals[:,:,-3])/evals[:,:,-1] #(k-aux+1,num_pts)
scattering = evals[:,:,-3]/evals[:,:,-1] #(k-aux+1,num_pts)
stack = np.stack((linearity, planarity, scattering), axis = -1) #(k-aux+1,num_pts)
dim_ent = entropy(stack) #(k-aux+1,num_pts)
k_opt = aux + np.argmin(dim_ent, axis = 0)


F = open("ConductorDimEnt.tex", "w+")




multiline = """
\\begin{{frame}}
\\begin{{figure}}[!h]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{}}}
\\end{{figure}}
\\end{{frame}}
"""

preamble = """
\\documentclass[pdf]{{beamer}}
\\setbeamertemplate{{navigation symbols}}{{}}
\\usefonttheme{{serif}}

\\mode<presentation>{{}}

\\title{{\\textbf{{}}}}
\\subtitle{{Dimensional entropy behaviour}}
\\author{{Samuel Dean}}
\\begin{{document}}
\\begin{{frame}}
\\titlepage
\\end{{frame}}
"""

F.write(preamble.format("Conductor"))

for j in [0]:
	fig = plt.figure()
	plt.bar(range(k-aux+1), dim_ent[:,j], align='center', alpha=0.5)
	plt.ylim(0.0,1.0)
	plt.xticks([0]+[10*(i+1)-2 for i in range(9)], [2]+[10*(i+1) for i in range(9)]) #(range(k-aux+2), range(aux,k+1))
	plt.xlabel('k')
	plt.ylabel('Dim. ent.')
	plt.title("Conductor")
	plotname = 'ConductorDimEntropy'+'.png'
	fig.savefig(plotname)
	plt.close()

	if j == 0:
		F.write(multiline.format("Conductor"))


for i in range(aux,k):
	for j in [0]:
		fig = plt.figure()
		y_pos = range(3)
		plt.bar(y_pos, stack[i-aux, j, :], align='center', alpha=0.5)
		plt.ylim(0.0,1.0)
		plt.xticks(y_pos, ["linearity","planarity","scattering"])
		plt.xlabel('States')
		plt.ylabel('Probability')
		plt.title("Conductor"+' k = '+str(i)+' entropy = '+str(dim_ent[i-aux,j]))
		plotname = 'ConductorPtDimEnt'+'k'+str(i)+'.png'
		fig.savefig(plotname)
		plt.close()

		if j == 0:
			F.write(multiline.format(plotname))

F.write("\\end{{document}}")
