""" Mealpy Example without precedence constraints:

https://github.com/thieu1995/mealpy/blob/master/examples/applications/discrete-problems/job_shop_scheduling.py

"""
import os
import numpy as np
from mealpy import PermutationVar, WOA, Problem
from mealpy import ACOR

# (job_id, machine_id)
job_times = [[2, 1, 3],
             [4, 2, 1],
             [3, 3, 2]]


# machine_times = [[3, 2, 1], [1, 4, 2], [2, 3, 2]]

n_jobs = 300
n_machines = 30
job_times = np.random.randint(1, 10, (n_jobs, n_machines))
# n_machines = len(machine_times)

data = {
    "job_times": job_times,
    # "machine_times": machine_times,
    "n_jobs": n_jobs,
    "n_machines": n_machines
}


class JobShopProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["per_var"]
        # makespan matrix with dim (n_job, m_machines)
        makespan = np.zeros((self.data["n_jobs"], self.data["n_machines"]))
        for gene in x:
            job_idx = gene // self.data["n_machines"]
            machine_idx = gene % self.data["n_machines"]
            # using DP to update the makespan matrix
            if job_idx == 0 and machine_idx == 0:
                makespan[job_idx][machine_idx] = job_times[job_idx][machine_idx]
            elif job_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx][machine_idx - 1] + job_times[job_idx][machine_idx]
            elif machine_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx - 1][machine_idx] + job_times[job_idx][machine_idx]
            else:
                makespan[job_idx][machine_idx] = max(makespan[job_idx][machine_idx - 1],
                                                     makespan[job_idx - 1][machine_idx]) + job_times[job_idx][machine_idx]
        return np.max(makespan)


bounds = PermutationVar(valid_set=list(range(0, n_jobs * n_machines)), name="per_var")
problem = JobShopProblem(bounds=bounds, minmax="min", data=data)

# model = WOA.OriginalWOA(epoch=100, pop_size=20)
# NOTE: change from WOA to ACOR
model = ACOR.OriginalACOR(epoch=1000, pop_size=100)
model.solve(problem, mode="process", n_workers=os.cpu_count() - 1)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")
