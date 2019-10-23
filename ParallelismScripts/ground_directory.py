import os
from laspy.file import File

condition = lambda x: ".las" in x and x[:4]=="TILE"

ground_command = "pdal ground initial_distance 0.5 -i {} -o {}"


json =	{
	 "type":"filters.range",
	"limits":"Classification[2:2]"
	}

for tile in os.listdir():
	if condition(tile)
		command = ground_command.format(tile, "ground_"+tile)
		os.system(ground_command)
		inFile = File("ground_"+tile, mode = "rw")
		ouFile = File("notgrd_"+tile, mode = "w")
		points = inFile.points
		ground = inFile.classification == 2
		inFile.points = points[ground]
		inFile.close()
		ouFile.points = points[~ ground]
		ouFile.close()


