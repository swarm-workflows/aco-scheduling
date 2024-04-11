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

mat_data = create_random_job_scheduling_problem(n_jobs=300, n_machines=10, seed=1)
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


def two_opt_swap(route, i, k):
    """
    Performs a 2-opt Swap: reverses the order of the jobs between two indices in the route.
    
    Parameters:
    route (list): The current route (job sequence).
    i (int): Start index of the portion of the route to be reversed.
    k (int): End index of the portion of the route to be reversed.
    
    Returns:
    list: The new route with the specified segment reversed.
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def calculate_makespan(job_sequence, job_times):
    """
    Calculate the makespan of a given job sequence.
    
    Parameters:
    job_sequence (list): The sequence of jobs.
    job_times (np.array): Matrix of job times with shape (n_jobs, n_machines).
    
    Returns:
    float: The makespan of the job sequence.
    """
    n_machines = job_times.shape[1]
    machine_times = np.zeros(n_machines)  # Total time for each machine
    
    for machine_idx, job_idx in enumerate(job_sequence):
        machine_idx = machine_idx % n_machines
        machine_times[machine_idx] += job_times[job_idx, machine_idx]
    
    makespan = np.max(machine_times)
    return makespan

def two_opt_local_search(initial_route, job_times, max_iterations=100):
    """
    Improves an existing route using the 2-opt algorithm.
    
    Parameters:
    initial_route (list): The initial route (job sequence).
    job_times (np.array): Matrix of job times.
    max_iterations (int): The maximum number of iterations to perform.
    
    Returns:
    list: The improved job sequence.
    """
    best_route = initial_route
    best_makespan = calculate_makespan(best_route, job_times)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, len(best_route) - 1):
            for k in range(i+1, len(best_route)):
                new_route = two_opt_swap(best_route, i, k)
                new_makespan = calculate_makespan(new_route, job_times)
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_route = new_route
                    improved = True
        iteration += 1
    
    return best_route


bounds = PermutationVar(valid_set=list(range(0, n_jobs)), name="per_var")
problem = JobShopProblem(bounds=bounds, minmax="min", data=data)


if 1:
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


    # After finding a solution with PSO
    x_decoded = model.problem.decode_solution(model.g_best.solution)
    initial_route = x_decoded["per_var"]
    # Perform 2-opt local search to improve the solution
    improved_route = two_opt_local_search(initial_route, data['job_times'])
    # Visualize and analyze the improved route
    fitness = calculate_makespan(improved_route, data["job_times"])
    visualize(data, improved_route, path='schedule-pso-ls.png', label=f"PSO+2-opt makespan: {fitness}")


if 1: 
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


    # After finding a solution with ACO
    x_decoded = model.problem.decode_solution(model.g_best.solution)
    initial_route = x_decoded["per_var"]
    # Perform 2-opt local search to improve the solution
    improved_route = two_opt_local_search(initial_route, data['job_times'])
    # Visualize and analyze the improved route
    fitness = calculate_makespan(improved_route, data["job_times"])
    visualize(data, improved_route, path='schedule-aco-ls.png', label=f"ACO+2-opt makespan: {fitness}")

