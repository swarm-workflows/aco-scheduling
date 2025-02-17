import collections

from ortools.sat.python import cp_model


def ortools_api(jobs, machines, **kwargs):
    r""" Solve the problem in OR-Tools

    Args:
        jobs (list[list[int]]): list of jobs with durations
        machines (list[list[int]]): list of machines for each task

    Notes:
     - jobs are the processing time for each task
     - machines are the machine id for each task, starting from 1

    Returns:
        solver (cp_model.CpSolver): OR-Tools solver object
    """

    # process jobs and machines
    n_jobs, n_machines = jobs.shape
    # compute horizon dynamically as the sum of all durations
    horizon = jobs.sum()
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple(
        "task_type", "start end interval"
    )
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            machine = machines[job_id][task_id]
            duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # create and add disjunctive constraints
    for machine in range(n_machines):
        model.AddNoOverlap(machine_to_intervals[machine])

    # precedences inside a job
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs)],
    )
    model.minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs):
            for task_id, task in enumerate(job):
                machine = machines[job_id][task_id]
                duration = jobs[job_id][task_id]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=duration,
                    )
                )

        # Create per machine output lines.
        output = ""
        for machine in range(n_machines):
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.objective_value}")
        print(output)
    else:
        print("No solution found.")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")
    return solver
