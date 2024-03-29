import numpy as np
from mealpy import PermutationVar, WOA, ACOR, PSO, Problem
import matplotlib.pyplot as plt
import sys


def create_random_job_scheduling_problem(n_jobs, n_machines, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mat_data = np.random.triangular(10, 150, 300, size=n_jobs*n_machines)
    #mat_data = np.random.normal((n_jobs*n_machines)) * 50 + 50  # Random processing times
    #mat_data = np.random.normal((n_jobs*n_machines)) * 50 + 50  # Random processing times
    
    #mat_data = np.random.random((n_jobs,n_machines)) * 100 
    mat_data = mat_data.reshape(n_jobs,n_machines)
    #sys.exit(0)
    return(mat_data)

mat_data = create_random_job_scheduling_problem(n_jobs=3000, n_machines=1000, seed=1)
job_times = mat_data 
print(job_times.shape)

n_jobs = job_times.shape[0]
n_machines = job_times.shape[1]

data = {
    "job_times": job_times,
    "n_jobs": n_jobs,
    "n_machines": n_machines
}


def visualize(data, x, path=None, label=None):
    """
    Visualization for job scheduling problem.
    """
    print(x)
    time_mat = data['job_times']
    print(time_mat)
    #sys.exit(0)
    with plt.style.context('ggplot'):
        n_machines, n_jobs = data['n_machines'], data['n_jobs'], 

        # Compute total times and job starts for visualization
        machine_times = np.zeros(n_machines)
        job_starts = {i: [] for i in range(n_machines)}  # Store start time for each job on each machine
        for  machine_idx, job_idx in enumerate(x):
            #print('----------------------')
            #machine_idx = int(machine_idx)
            machine_idx = int(machine_idx) % (n_machines)  # Apply modulo operation
            start_time = machine_times[machine_idx]
            #print(machine_idx)
            #print(machine_idx,job_idx)
            #print(start_time)
            job_starts[machine_idx].append(start_time)
            #print(machine_idx,job_idx)
            machine_times[machine_idx] += time_mat[job_idx][machine_idx]
            #print(machine_times[machine_idx])
            #print('----------------------')

        #print(x)
        #print(job_starts)
        fig, ax = plt.subplots()
        Y = np.arange(n_machines)

        # Create bars for the Gantt chart
        for machine_idx in range(n_machines):
            for i, start_time in enumerate(job_starts[machine_idx]):
                ax.barh(machine_idx, machine_times[machine_idx] - start_time, left=start_time, height=0.5, label=f"Job {i}", align='center')

        # Set labels and titles
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_yticks(Y)
        ax.set_yticklabels([f"M{i}" for i in Y])
        ax.set_title("Job Scheduling: %s" % label)
        plt.tight_layout()
        if path is not None:
            plt.savefig(path)
        plt.show()


class JobShopProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["per_var"]
        machine_times = np.zeros(self.data["n_machines"])  # Total time for each machine
        for machine_idx, job_idx in enumerate(x):
            machine_idx = int(machine_idx) % (self.data["n_machines"])  # Apply modulo operation
            machine_times[machine_idx] += job_times[job_idx][machine_idx]
        makespan = np.max(machine_times)
        return np.max(makespan)



bounds = PermutationVar(valid_set=list(range(0, n_jobs)), name="per_var")
problem = JobShopProblem(bounds=bounds, minmax="min", data=data)


model = PSO.AIW_PSO(epoch=100, pop_size=100, seed=10)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
x_decoded = model.problem.decode_solution(model.g_best.solution)
x = x_decoded["per_var"]
visualize(data, x, path='schedule-pso.png',label=f"PSO makespan: {model.g_best.target.fitness}")
model.history.save_global_objectives_chart(filename="goc-pso")
model.history.save_local_objectives_chart(filename="loc-pso")

model = ACOR.OriginalACOR(epoch=100, pop_size=100, seed=10)
model.solve(problem)
print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
x_decoded = model.problem.decode_solution(model.g_best.solution)
x = x_decoded["per_var"]
visualize(data, x, path='schedule-aco.png',label=f"ACO makespan: {model.g_best.target.fitness}")
model.history.save_global_objectives_chart(filename="goc-aco")
model.history.save_local_objectives_chart(filename="loc-aco")
