from multiprocessing import cpu_count, Process

def splitter(job):

	def proceed(N, processes):
		pool = {i: processes.pop() for i in range(N)}
		for item in pool:
			pool[item].start()
		for item in pool:
			pool[item].join()
		pool.clear()

	def split_job(tile_list):
		processes = list([Process(target=job, args=(tile,)) for tile in tile_list])
		num_cores = cpu_count()
		num_processes = len(processes)
		proceed(num_processes%num_cores, processes)
		for step in range(num_processes//num_cores):
			proceed(num_cores, processes)

	return split_job