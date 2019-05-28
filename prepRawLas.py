import os
import lasmaster as la
for filename in os.listdir():
	if filename[:4]=="T200":
		la.attr(la.nfl(filename))
