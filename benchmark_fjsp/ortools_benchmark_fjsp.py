""" Example from: https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py """
import collections
import random

from ortools.sat.python import cp_model
from utils import read_fjsp_file


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    r"""Print intermediate solutions."""

    def __init__(self) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        """Called at each new solution."""
        print(f"Solution {self.__solution_count}, time = {self.wall_time} s,"
              f" objective = {self.objective_value}")
        self.__solution_count += 1


def flexible_jobshop(jobs) -> None:
    r"""Flexible job-shop problem.

    Args:
        jobs (list): List of jobs. Each job is a list of tasks. Each task is a list of tuples (duration, machine).

    Returns:
        None
    """
    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    num_machines = 1000
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    # upper bound: sum of all tasks' durations
    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    print(f"Horizon = {horizon}")

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends: list[cp_model.IntVar] = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = f"_j{job_id}_t{task_id}"
            start = model.new_int_var(0, horizon, "start" + suffix_name)
            duration = model.new_int_var(
                min_duration, max_duration, "duration" + suffix_name
            )
            end = model.new_int_var(0, horizon, "end" + suffix_name)
            interval = model.new_interval_var(
                start, duration, end, "interval" + suffix_name
            )

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = f"_j{job_id}_t{task_id}_a{alt_id}"
                    l_presence = model.new_bool_var("presence" + alt_suffix)
                    l_start = model.new_int_var(0, horizon, "start" + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.new_int_var(0, horizon, "end" + alt_suffix)
                    l_interval = model.new_optional_interval_var(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                    )
                    l_presences.append(l_presence)

                    # Link the primary/global variables with the local ones.
                    model.add(start == l_start).only_enforce_if(l_presence)
                    model.add(duration == l_duration).only_enforce_if(l_presence)
                    model.add(end == l_end).only_enforce_if(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.add_exactly_one(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.new_constant(1)

        if previous_end is not None:
            job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.add_no_overlap(intervals)

    # Makespan objective
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, job_ends)
    model.minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()
    status = solver.solve(model, solution_printer)

    # Print final solution.
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Optimal objective value: {solver.objective_value}")
        for job_id in all_jobs:
            print(f"Job {job_id}")
            for task_id, task in enumerate(jobs[job_id]):
                start_value = solver.value(starts[(job_id, task_id)])
                machine: int = -1
                task_duration: int = -1
                selected: int = -1
                for alt_id, alt in enumerate(task):
                    if solver.boolean_value(presences[(job_id, task_id, alt_id)]):
                        task_duration, machine = alt
                        selected = alt_id
                print(f"  task_{job_id}_{task_id} starts at {start_value} (alt"
                      f" {selected}, machine {machine}, duration {task_duration})")

    print(solver.response_stats())


def random_case() -> None:
    """solve a random flexible jobshop problem."""
    # Data part.
    num_jobs = 50
    num_machines = 1000
    max_tasks_per_job = num_jobs
    max_alternatives_per_task = num_jobs
    max_duration = 100

    jobs = []
    for _ in range(num_jobs):
        num_tasks = random.randint(1, max_tasks_per_job)
        job = []
        for _ in range(num_tasks):
            num_alternatives = random.randint(1, max_alternatives_per_task)
            task = []
            for _ in range(num_alternatives):
                duration = random.randint(1, max_duration)
                machine = random.randint(0, num_machines - 1)
                task.append((duration, machine))
            job.append(task)
        jobs.append(job)

    flexible_jobshop(jobs)


if __name__ == "__main__":
    # random_case()
    # TODO: use argparser
    fn = "./Monaldo/Fjsp/Job_Data/Brandimarte_Data/Text/Mk06.fjs"
    # jobs = read_file(fn)
    num_jobs, num_machines, jobs = read_fjsp_file(fn)
    flexible_jobshop(jobs)
    # NOTE: terminate with Ctrl+C if it's running for too long
