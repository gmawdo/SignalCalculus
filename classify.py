import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import datetime
from laspy.file import File
import os

for lasfile in (filename for filename in os.listdir("OLD-TEST-FILES/TestFiles") if filename[:4]=="attr"):

	inFile = File("OLD-TEST-FILES/TestFiles/"+lasfile, mode = "r")

	print(lasfile)
	print(max(inFile.z))
	# point here is part of pylon I tjhink : 385247.370 207848.540 80.070
	#definitely part of a pylon here: 385242.410 207840.860 79.200
	#part of an obscured pylon - 385288.360 207914.330 82.310
	
	#ISO = inFile.iso
	#LANG = inFile.lang
	#pointCuboid = (inFile.x>385240)  & (inFile.x<385300) & (inFile.y>207840) & (inFile.y<207900) & (inFile.z<80) & (inFile.iso > 0.6) & (inFile.lang < 0.1)
	#isoPoints = ISO[pointCuboid] #points in area of pylon
	#iso1,isoc1 = np.unique(isoPoints, axis=0,return_counts=True)
	#langPoints = LANG[pointCuboid] #points in area of pylon
	#lang1,langc1 = np.unique(langPoints, axis=0,return_counts=True)
	
	#I think we need to plot the above values, so....
	#plt.plot(iso1,isoc1) # plotting by columns
	#lots of points with iso less than 0.6 but these are not the ones, in fact the opposite!
	#plt.show()
	#plt.plot(lang1,langc1) # plotting by columns
	#gives fairly even distribution
	#plt.show()
	
	# inFile.x
	# create matrix of all the eigenvectors
	#print(max(np.round(inFile.lang,0)),min(np.round(inFile.lang)))
	result1 = np.stack((np.round(inFile.iso,3),np.round(inFile.lang,2)), axis=-1)
	print("formed main result set size:",len(result1))
	print(result1.shape)
	
	# find how many unique isotropies in combo with linear angles
	u1,i1,c1 = np.unique(result1, axis=0,return_inverse=True,return_counts=True)
	c1 = c1.reshape(u1.shape[0],1)
	u1 = np.concatenate((u1,c1),axis=1)
	#condition = (u1[:,2]>20) & (u1[:,2]<750)
	#condition = (u1[:,1]<0.2)*(u1[:,2]>50)
	#u1 = u1[condition]
	u1.sort
	print("form unqiue result set of distinct isotropies and langs combos with counts",u1.shape)
	print(u1.shape) #total number of tuples
	
	# for info purposes get unqiue number of isotropies and unique number of linear angles
	u2,i2,c2 = np.unique(u1[:,[0]], axis=0,return_inverse=True,return_counts=True)
	u3,i3,c3 = np.unique(u1[:,[1]], axis=0,return_inverse=True,return_counts=True)
	print("number of distinct isotropies:",u2.shape)
	u2.sort
	print("number of distinct linear angles:",u3.shape)
	
	#isotropies,linear angle, cnt
	x, y, z = u1[:,0], u1[:,1], u1[:,2]
	
	#classify a set of points into a new file
	outFile = File("CLASSIFIED/C-"+lasfile[:-4]+".las", mode = "w", header = inFile.header)
	classification = inFile.classification
	outFile.points = inFile.points
	
	#writecondition = (inFile.ent==0.000000001)
	#classification[np.logical_not(writecondition)]=0
	classification *= 0 #initialise and classes to zero 
	
	#uprights
	#(inFile.z<85) &
	pointCuboid2 =   (inFile.iso >= 0.6) & (inFile.iso < 0.7) & (inFile.lang < 0.1)
	classification[pointCuboid2]=11
	#(inFile.z<85) & 
	pointCuboid2 =  (inFile.iso >= 0.7) & (inFile.iso < 0.8) & (inFile.lang < 0.1)
	classification[pointCuboid2]=12
	#pointCuboid2 =  (inFile.z<85) & (inFile.iso >= 0.8) & (inFile.iso < 0.9) & (inFile.lang < 0.1)
	#classification[pointCuboid2]=13
	#pointCuboid2 =  (inFile.z<85) & (inFile.iso >= 0.5) & (inFile.iso < 0.6) & (inFile.lang < 0.1)
	#classification[pointCuboid2]=14
	
	#noise = (np.round(inFile.lang,3)<=0.003) # & (inFile.z > 80)
	#classification[noise]=13
	
	#(inFile.z<71) & 
	#ground = (inFile.iso >= 0.8) & (inFile.iso <= 0.87) & (inFile.lang >= 0.94) & (inFile.lang <= 0.96) 
	#classification[ground]=14
	
	#conductor
	writecondition_conductor = (np.round(inFile.iso,3)>0.5) & (np.round(inFile.iso,3)<0.6) & (np.round(inFile.lang,2)>0.9) # & (np.round(inFile.z,0) < 85)
	classification[writecondition_conductor]=10
	
	outFile.classification = classification
	outFile.close()
	
	inFile.close()
	#PLOT
	#PLOT
	fig = plt.figure()
	
	#ax = fig.gca(projection='3d')
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=16, azim=43)
	
	ax.set_xlabel('isotropy')
	ax.set_ylabel('lang')
	ax.set_zlabel('num points')
	
	ax.set_xlim3d(0.0, 1.0) #iso
	ax.set_ylim3d(0.0, 1.0) #lang
	ax.set_zlim3d(0, 10000) #numpoints
	
	#ax.set_xlim3d(0, 200)
	#ax.set_ylim3d(0.0, 0.03)
	#ax.set_zlim3d(0, 500)
	verts = []
	zs = u3[:,0]
	facecolors =[]
	for z in u3[:,0]: #for each linear angle
	    #print(z)
	    ux = u1[(u1[:,1]==z)] #this predicate extracts entries only for this lang value
	    xs = np.concatenate([[0],ux[:,0],[0]]) #iso
	    ys = np.concatenate([[0],ux[:,2],[0]]) #num points
	    #print(xs)
	    #print("****")
	    #print(ys)
	    verts.append(list(zip(xs,ys)))
	    facecolors.append((z, z * z, 0.0, 0.6))
	
	poly = PolyCollection(verts, facecolors=facecolors)
	poly.set_alpha(0.7)
	ax.add_collection3d(poly, zs=zs, zdir='y')
	
	#X, Y = np.meshgrid(x, y, sparse=True)
	#Z = z.reshape(X.shape)
	#ax.plot_surface(X, Y, Z)
	
	#plt.show()
	plt.savefig('CHARTS/plot'+lasfile[:-4]+'.png')
