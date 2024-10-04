import numpy as np
from collections import defaultdict


class MaxMinAntSystemJobScheduling:
    r"""

    Reference:
     - Stützle, Thomas, and Holger H. Hoos. "MAX-MIN ant system." Future generation computer systems 16.8 (2000): 889-914.
    """

    def __init__(self, num_ants, num_jobs, num_machines, alpha, beta, rho, tau_min, tau_max, max_iterations):
        self.num_ants = num_ants
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_iterations = max_iterations
        self.pheromone = np.full((num_jobs, num_machines), tau_max)
        self.best_solution = None
        self.best_cost = float('inf')

    def initialize_processing_times(self, processing_times):
        r""" Initialize the processing times for each job and operation on each machine.

        Args:
            processing_times (List[List[int]]): A list of lists where each inner list contains the processing times for
                each operation of a job on each machine.
        """
        self.processing_times = processing_times

    def initialize_pheromone(self):
        r""" Initialize the pheromone matrix with the maximum value."""
        self.pheromone.fill(self.tau_max)

    def run(self):
        r""" Run the MAX-MIN Ant System algorithm.

        Returns:
            Tuple[Dict[int, List[Tuple[int, int, int, int]]], float]: A tuple containing the best solution and its cost.
        """
        for iteration in range(self.max_iterations):
            solutions = self.construct_solutions()
            costs = [self.calculate_cost(solution) for solution in solutions]
            best_ant = np.argmin(costs)
            best_ant_cost = costs[best_ant]
            best_ant_solution = solutions[best_ant]
            if best_ant_cost < self.best_cost:
                self.best_cost = best_ant_cost
                self.best_solution = best_ant_solution
            self.update_pheromones(best_ant_solution)
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Cost: {self.best_cost}")
            # TODO: Implement a stagnation detection mechanism
            if self.detect_stagnation():
                self.initialize_pheromone()
        return self.best_solution, self.best_cost

    def construct_solutions(self):
        r""" Construct solutions using the MAX-MIN Ant System algorithm, record each solution from ants.

        Returns:
            List[Dict[int, List[Tuple[int, int, int, int]]]]: A list of solutions where each solution is a dictionary
                containing the machine id as the key and a list of tuples containing the job id, operation id, start time,
                and finish time.
        """
        solutions = []
        for _ in range(self.num_ants):
            solution = self.construct_solution()
            solutions.append(solution)
        return solutions

    def construct_solution(self):
        r""" Construct a solution using the MAX-MIN Ant System algorithm.

        Returns:
            Dict[int, List[Tuple[int, int, int, int]]]: A dictionary containing the machine id as the key and a list of
                tuples containing the job id, operation id, start time, and finish time.
        """
        solution = defaultdict(list)
        unassigned_jobs = list(range(self.num_jobs))
        machine_available_time = [0] * self.num_machines
        while unassigned_jobs:
            job = np.random.choice(unassigned_jobs)
            for operation in range(len(self.processing_times[job])):
                probabilities = self.calculate_probabilities(job, machine_available_time)
                machine = np.random.choice(list(range(self.num_machines)), p=probabilities)
                start_time = machine_available_time[machine]
                finish_time = start_time + self.processing_times[job][operation]
                solution[machine].append((job, operation, start_time, finish_time))
                machine_available_time[machine] = finish_time
            unassigned_jobs.remove(job)
        return solution

    def calculate_probabilities(self, job, machine_available_time):
        r""" Calculate the probabilities of assigning a job to a machine.

        .. math::
            p_{ij} = \frac{(\tau_{ij})^{\alpha} \cdot (\eta_{ij})^{\beta}}{\sum_{k=1}^{m} (\tau_{ik})^{\alpha} \cdot (\eta_{ik})^{\beta}}

        Args:
            job (int): The job id.
            machine_available_time (List[int]): The available time on each machine.

        Returns:
            np.ndarray: The probabilities of assigning the job to each machine.
        """
        pheromone = self.pheromone[job]
        heuristic = 1.0 / (np.array(machine_available_time) + 1)
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_cost(self, solution):
        r""" Calculate the makespan of a solution.

        Args:
            solution (Dict[int, List[Tuple[int, int, int, int]]]): The solution to evaluate.

        Returns:
            int: The makespan of the solution.
        """
        makespan = 0
        for machine, tasks in solution.items():
            finish_time = max(task[3] for task in tasks)
            makespan = max(makespan, finish_time)
        return makespan

    def update_pheromones(self, best_solution):
        r""" update the pheromones based on the best solution found by the ants.

        .. math::
            \tau_{ij} = (1 - \rho) \cdot \tau_{ij} + \sum_{k=1}^{m} \Delta \tau_{ij}^{k}

        Args:
            best_solution (Dict[int, List[Tuple[int, int, int, int]]]): The best solution found by the ants.
        """
        self.pheromone *= (1 - self.rho)
        for machine, tasks in best_solution.items():
            for job, operation, start_time, finish_time in tasks:
                self.pheromone[job][machine] += 1.0 / self.best_cost
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def detect_stagnation(self):
        # TODO: Implement a stagnation detection mechanism
        return False


# Example usage
num_jobs = 5
num_machines = 3
num_ants = 10
alpha = 1.0
beta = 2.0
rho = 0.1
tau_min = 0.1
tau_max = 5.0
max_iterations = 50
processing_times = [
    [3, 2, 2],  # Job 0: operation 1 on machine 0 takes 3, operation 2 on machine 1 takes 2, etc.
    [2, 3, 1],
    [4, 3, 2],
    [2, 4, 3],
    [3, 1, 2]
]
mmas = MaxMinAntSystemJobScheduling(
    num_ants,
    num_jobs,
    num_machines,
    alpha,
    beta,
    rho,
    tau_min,
    tau_max,
    max_iterations)
mmas.initialize_processing_times(processing_times)
best_solution, best_cost = mmas.run()
print(f"Best Solution: {best_solution}")
print(f"Best Cost: {best_cost}")
