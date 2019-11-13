import os
import DEMO_pack as dp

os.chdir("TILES")

tiles = [t for t in os.listdir() if t[:3]=="DH5"]
print(f"Adding pt ID to {len(tiles)} files.")
dp.functions.add_pid(tiles)

tiles = [t for t in os.listdir() if t[:3]=="ID_"]
print(f"Classifying ground on to {len(tiles)} files.")
dp.functions.classify_ground(tiles)

tiles = [t for t in os.listdir() if t[:7]=="ground_"]
print(f"Adding pt ID to {len(tiles)} files.")
dp.functions.ground_removal(tiles)

tiles = [t for t in os.listdir() if t[:3]=="not"]
print(f"Adding attributes to {len(tiles)} files.")
dp.functions.add_attributes(tiles)

tiles = [t for t in os.listdir() if t[:4]=="attr"]
print(f"Adding classification to {len(tiles)} files.")
dp.functions.add_classification(tiles)

tiles = [t for t in os.listdir() if t[:3] == "grd"]
print(f"Reapplying ground to {len(tiles)} files.")
pairs = []
for t in tiles:
	for s in os.listdir():
		if t in s and s[:10]=="classified":
			pairs.append((s, t))
dp.functions.zip(pairs)

tiles = [t for t in os.listdir() if t[:4]=="zipp"]
print(f"Adding HAG to {len(tiles)} files.")
dp.functions.add_hag(tiles, 1, 0.01)

tiles = [t for t in os.listdir() if t[:3]=="hag"]
print(f"Finding buildings and ground on {len(tiles)} files.")
dp.functions.bbox(tiles)

tiles = [t for t in os.listdir() if t[:3]=="bb_"]
print(f"Finding conductors on {len(tiles)} files.")
dp.functions.conductor_matters(tiles)

tops = []
for t in os.listdir():
	if t[:3]=="cm_":
		for s in os.listdir():
			if s in t and s[:3]=="DH5":
				tops.append((t,s))

print(f"Finalising results on {len(tiles)} files.")
dp.functions.finisher(tops)