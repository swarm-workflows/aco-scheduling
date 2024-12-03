import os
from glob import glob


def read_fjsp_file(fn):
    r""" Read the FJSP file.

    Args:
        fn (str): Filename of case.

    Returns:
        tuple (int, int, list): Number of jobs, number of machines, and list of jobs.
        Each job is a list of operations.
        Each operation is a list of tuples (machine, processing_time).

    Note:
        - Detailed description of the FJSP format can be found at [DataSetExplanation.txt](./DataSetExplanation.txt)
    """
    # read file to lines
    try:
        with open(fn, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fn}")

    # First line: number of jobs and machines
    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    jobs = []

    # Parse each job line
    for line in lines[1:]:
        # Note: `Mk` cases have empty lines
        if line.strip() == '':
            continue
        job_data = line.strip().split()
        num_operations = int(job_data[0])
        operations = []
        index = 1
        for _ in range(num_operations):
            k = int(job_data[index])
            index += 1
            machines = []
            for _ in range(k):
                machine = int(job_data[index])
                processing_time = int(job_data[index + 1])
                machines.append((machine, processing_time))
                index += 2
            operations.append(machines)
        jobs.append(operations)

    return num_jobs, num_machines, jobs


def read_all_fjsp_files(folder_path):
    r""" Read all FJSP files

    Args:
        folder_path (str): Folder path containing all FJSP files.

    Returns:
        dict: Dictionary of all jobs data.
            Key: Filename of the FJSP file.
            Value: Dictionary containing number of jobs, number of machines, and list of jobs.
            Each job is a list of operations.
            Each operation is a list of tuples (machine, processing_time).
    """
    # read all .fjs files
    fjsp_files = glob(os.path.join(folder_path, '**', '*.fjs'), recursive=True)
    all_jobs_data = {}
    for fjsp_file in fjsp_files:
        num_jobs, num_machines, jobs = read_fjsp_file(fjsp_file)
        # key: file, value: dict
        all_jobs_data[fjsp_file] = {
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'jobs': jobs
        }
    return all_jobs_data
