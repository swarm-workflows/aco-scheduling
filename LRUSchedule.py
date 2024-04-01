import heapq
from time import time
import numpy as np

class LRUSchedule:
    def __init__(self, data, order_by_time=None):
        self.n_machines = data["n_machines"]
        self.n_jobs = data["n_jobs"]
        self.job_times = data["job_times"]
        
        self.jobs = [[np.mean(data["job_times"][i]), np.median(data["job_times"][i]), i] for i in range(self.n_jobs)]
        if order_by_time == 'mean':
            ####decreasing order based on avg time, if equal time increasing order of job id####
            self.jobs.sort(key = lambda x: (-x[0], x[2]))
        elif order_by_time == 'median':
            ####decreasing order based on avg time, if equal time increasing order of job id####
            self.jobs.sort(key = lambda x: (-x[1], x[2]))

        self.solution = {i: {"jobs": [], "total_time": 0} for i in range(self.n_machines)}
        self.makespan = -1
        

    def solve(self):
        start_t = time()
        
        #initialized lru with all machines to 0 total time
        self.makespan = 0
        lru_q = [[0, i] for i in range(self.n_machines)]
        heapq.heapify(lru_q)

        for [avg_time, med_time, j] in self.jobs:
            m = heapq.heappop(lru_q)[1]
            self.solution[m]["jobs"].append([self.solution[m]["total_time"], j])
            self.solution[m]["total_time"] += self.job_times[j][m]
            self.makespan = max(self.makespan, self.solution[m]["total_time"])
            heapq.heappush(lru_q, [self.solution[m]["total_time"], m])

        end_t = time()
        print(f"Time to run the LRU algorithm: {end_t - start_t} seconds")

        
    def getSolution(self):
        return self.solution

    def getMakespan(self):
        return self.makespan
 

class OptimizedLRUSchedule:
    def __init__(self, data, sortest_first=True):
        self.n_machines = data["n_machines"]
        self.n_jobs = data["n_jobs"]
        self.job_times = data["job_times"]
        
        self.machine_job_times = {i: {"jobs": [], "last_i": 0} for i in range(self.n_machines)}
        for j in range(self.n_jobs):
            for m in range(self.n_machines):
                self.machine_job_times[m]["jobs"].append([self.job_times[j][m], j])

        for m in range(self.n_machines):
            if sortest_first:
                self.machine_job_times[m]["jobs"].sort(key = lambda x: x[0])
            else:
                self.machine_job_times[m]["jobs"].sort(key = lambda x: -x[0])

        self.solution = {i: {"jobs": [], "total_time": 0} for i in range(self.n_machines)}
        self.makespan = -1
        

    def solve(self):
        start_t = time()
        
        #initialized lru with all machines to 0 total time
        self.makespan = 0
        lru_q = [[0, i] for i in range(self.n_machines)]
        heapq.heapify(lru_q)

        seen = set()

        while len(seen) < self.n_jobs:
            m = heapq.heappop(lru_q)[1]
            while self.machine_job_times[m]["last_i"] < self.n_jobs:
                e = self.machine_job_times[m]["jobs"][self.machine_job_times[m]["last_i"]]
                if e[1] in seen:
                    self.machine_job_times[m]["last_i"] += 1
                    continue
                else:
                    seen.add(e[1])
                    self.solution[m]["jobs"].append([self.solution[m]["total_time"], e[1]])
                    self.solution[m]["total_time"] += e[0]
                    self.makespan = max(self.makespan, self.solution[m]["total_time"])
                    self.machine_job_times[m]["last_i"] += 1
                    heapq.heappush(lru_q, [self.solution[m]["total_time"], m])
                    break

        end_t = time()
        print(f"Time to run the LRU algorithm: {end_t - start_t} seconds")

        
    def getSolution(self):
        return self.solution

    def getMakespan(self):
        return self.makespan
        
