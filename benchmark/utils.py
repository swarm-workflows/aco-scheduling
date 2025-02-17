import os.path as osp


def read_jssp_file(fn, problem="", id="", format="taillard"):
    r"""Read job file and return duration and machine matrices.

    Args:
        fn (str): File name.
        problem (str): Problem name.
        id (str): Problem ID.
        format (str): File format.

    Returns:
        tuple: Duration and machine matrices.
    """
    if not osp.exists(fn):
        raise FileExistsError(f"Error: Problem {problem}{id} does not exist.")
    durations = []
    machines = []

    if format == "taillard":
        with open(fn, 'r') as file:
            first_line = file.readline().strip()
            num_jobs, num_machines = map(int, first_line.split())

            for _ in range(num_jobs):
                line = file.readline().strip()
                durations.append(list(map(int, line.split())))

            for _ in range(num_jobs):
                line = file.readline().strip()
                machines.append(list(map(int, line.split())))
            # adjust machine id
            machines = [[m - 1 for m in machine] for machine in machines]
    else:
        with open(fn, 'r') as file:
            first_line = file.readline().strip()
            num_jobs, num_machines = map(int, first_line.split())

            for _ in range(num_jobs):
                line = file.readline().strip().split()
                job_machines = []
                job_durations = []
                for i in range(0, len(line), 2):
                    # Adjust machine ID by subtracting 1 for 0-based indexing
                    job_machines.append(int(line[i]))
                    job_durations.append(int(line[i + 1]))
                machines.append(job_machines)
                durations.append(job_durations)

    return durations, machines
