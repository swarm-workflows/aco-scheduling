import itertools
import json

import networkx as nx
import numpy as np


def convert_to_nx(times, machines, n_jobs, n_machines, **kwargs):
    r""" Convert input data to conjunctive graph and disjunctive graph.

    Args:
        times (np.ndarray): processing times of jobs on machines
        machines (np.ndarray): machine assignments of jobs
        n_jobs (int): number of jobs
        n_machines (int): number of machines

    Returns:
        dep_graph (nx.DiGraph): conjunctive graph
        res_graph (nx.Graph): disjunctive graph
    """
    if kwargs.get('verbose', False):
        print(f'jobs={n_jobs}, machines={n_machines}')
    # Conjunctive graph (directed)
    dep_graph = nx.DiGraph()

    dep_graph.add_node('s', duration=0)
    dep_graph.add_node('t', duration=0)
    for job, step in itertools.product(range(n_jobs), range(n_machines)):
        machine = machines[job][step]
        prev_machine = machines[job][step - 1] if step > 0 else None
        duration = times[job][step]
        # node in ConjGraph: (job, machine)
        dep_graph.add_node((job, machine), duration=duration)
        # edge in ConjGraph: (job, prev_machine) -> (job, machine)
        if prev_machine is not None:
            dep_graph.add_edge((job, prev_machine), (job, machine))
        else:
            # edge in ConjGraph: s -> (job, machine)
            dep_graph.add_edge('s', (job, machine))
        if step == n_machines - 1:
            dep_graph.add_edge((job, machine), 't')

    # Disjunctive graph (undirected)
    res_graph = nx.Graph()
    for job, machine in itertools.product(range(n_jobs), range(n_machines)):
        # node in DisjGraph: (job, machine), same as ConjGraph
        res_graph.add_node((job, machine))
    # edge in DisjGraph: (job1, machine) -- (job2, machine)
    for job1, job2, machine in itertools.product(range(n_jobs), range(n_jobs), range(n_machines)):
        if job1 < job2:
            res_graph.add_edge((job1, machine), (job2, machine))

    return dep_graph, res_graph


def load_network_fn(fn: str, sid: int, **kwargs):
    r""" Load network from file and return the graph.

    Args:
        fn (str): file name
        sid (int): position of the network

    Returns:
        tuple: conjunctive graph and disjunctive graph
    """
    data = np.load(fn)
    print(data.shape)
    times, machines = data[sid]
    print(times.shape)
    n_jobs, n_machines = data.shape[2:]
    return convert_to_nx(times, machines, n_jobs, n_machines)


def generate_random_machines(n, m):
    r""" Generate random machine assignments.

    Args:
        n (int): number of jobs
        m (int): number of machines
    """
    machines = np.zeros((n, m), dtype=int)
    for i in range(n):
        machines[i] = np.random.permutation(m) + 1
    return machines


def draw_networks(g1, g2=None, fn=None):
    r""" Draw the conjunctive graph and disjunctive graph.

    Args:
        g1 (nx.DiGraph): conjunctive graph
        g2 (nx.Graph): disjunctive graph
        fn (str, optional): file name. Default: None.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    fn = fn if fn else 'dgraph_aco'
    with plt.style.context('ggplot'):
        # pos = nx.spring_layout(g1, seed=7)
        pos = nx.kamada_kawai_layout(g1)
        nx.draw_networkx_nodes(g1, pos, node_size=7)
        nx.draw_networkx_labels(g1, pos)
        nx.draw_networkx_edges(g1, pos, arrows=True)
        if g2:
            nx.draw_networkx_edges(g2, pos, edge_color='r')
        plt.draw()
        plt.savefig(f"{fn}.png")
        # plt.show()


def plot_aco_vs_ortools(aco_best, ortools_best, fn=None):
    r""" Plot ACO vs OR-Tools.

    Args:
        aco_best (list): ACO best solutions
        ortools_best (float): OR-Tools best solution
        fn (str, optional): file name. Default: None.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    fn = fn if fn else 'ACO_vs_ORTools'
    with plt.style.context('ggplot'):
        fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
        axs.plot(np.arange(len(aco_best)), aco_best, label='ACO')
        axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Makespan')
        axs.legend()
        plt.savefig(f"{fn}.png")
        # plt.show()

def store(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)
