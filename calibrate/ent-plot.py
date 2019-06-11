import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File
from sklearn.neighbors import NearestNeighbors

inFile = File("ENEL/000/attrDF2000305_Completa.laz.las-GpsTime139236.86195908333333333332139256.48610712499999999998N006k050radius00_50thresh0_001v_speed02_00dec00_10.las", mode = "r")

# create matrix of all the eigenvectors
result1 = np.stack((inFile.impdec), axis=-1)
print("formed main result set size:",len(result1))
u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
print(u1.shape)

plt.plot(u1)
plt.show()