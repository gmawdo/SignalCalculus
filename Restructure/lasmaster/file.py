from laspy.file import File

# Governs the interaction with laspy

def std_data_types():
	output = 	{
			"p0"	:	(lambda x, y, z: x/(x+y+z)),
			"p1"	:	(lambda x, y, z: y/(x+y+z)),
			"p2"	:	(lambda x, y, z: z/(x+y+z)),
			"eig0"	:	(lambda x, y, z: x),
			"eig1"	:	(lambda x, y, z: y),
			"eig2"	:	(lambda x, y, z: z),
			"iso"	:	(lambda x, y, z: (x+y+z)/np.sqrt(3*(x**2+y**2+z**2))),
			"ent"	:	(lambda x, y, z: entropy(np.stack((x, y, z))/(x+y+z))),
			"rank"	:	(lambda x, y, z: 1*(x>thresh)+1*(y>thresh)+1*(z>thresh)),
			}

# U S E   L A S P Y   T O   T U R N   A   L A S   F I L E   I N T O
# A   D I C T I O N A R Y   O F   A T T R I B U T E S
# A N D   A   H E A D E R
def read_las(file_name):
	in_file = File(file_name, mode = "r")
	dimensions = [spec.name for spec in in_file.point_format]
	attributes = {}
	for dimension in attributes:
		dat = in_file.reader.get_dimension(dimension)
		attributes[dimension] = dat
	return (in_file.header, attributes)

# W R I T I N G   D A T A   T O   L A S
def write_las(name, header, attributes, data_types, descriptions):
	# create new dimensions	
	out_file = File(name, mode = "w", header = header)
	dimensions = [spec.name for spec in out_file.point_format]
	for dimension in attributes:
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = data_types[dimension], description = descriptions[dimension])

	# populate point records
	for dimension in attributes:
		dat = attributes[dimension]
		out_file.writer.set_dimension(dimension, dat)
