""" Mealpy example with precedence constraints:

* [x] add precedence constraints
* [x] build graph with conjunctive graph (nx.DiGraph) and disjunctive graph (nx.Graph)
* [x] add binary variables for disjunctive graph
* [x] check DAG
* [x] compute makespan
* [x] provide OR-Tools as baseline (A constraint programming solver, `pip install -U ortools`)
"""
import argparse
from matplotlib.pylab import f
import networkx as nx
from networkx import is_directed_acyclic_graph
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mealpy import PermutationVar, ACOR, PSO, Problem, BinaryVar


class DGProblem(Problem):
    def __init__(self, C: nx.DiGraph, D: nx.Graph):
        # conjunctive graph
        self.C = C
        # disjunctive graph
        self.D = D

        for node in D.nodes:
            assert C.has_node(node)
        self.enumerated_nodes = list(D.nodes)
        # NOTE: graph bounds as binary variables
        self.graph_bounds = BinaryVar(n_vars=len(D.edges), name='dir_var')

        super().__init__(bounds=self.graph_bounds, minmax='min', log_to='file')

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded['dir_var']
        return self.compute_makespan(x)

    def build_dag(self, x):
        r""" build dag from permutation.

        Args:
            x (list): permutation

        Returns:
            dag (nx.DiGraph): directed acyclic graph
        """
        dag = self.C.copy()
        assert isinstance(dag, nx.DiGraph)
        # for each edge in D, assign direction u->v if x[edge_idx] == 1, else v->u
        # check graph is acyclic per each operation, otherwise remove the edge
        for edge_idx, (u, v) in enumerate(self.D.edges()):
            if x[edge_idx] == 1:
                dag.add_edge(u, v)
                if nx.is_directed_acyclic_graph(dag):
                    continue
                else:
                    dag.remove_edge(u, v)
            else:
                dag.add_edge(v, u)
                if nx.is_directed_acyclic_graph(dag):
                    continue
                else:
                    dag.remove_edge(v, u)
        return dag

    def compute_makespan(self, x):
        r""" Compute the makespan of the given permutation.
        """
        dag = self.build_dag(x)
        # DEBUG: large graphs end with `inf` makespan
        if not nx.is_directed_acyclic_graph(dag):
            return np.inf

        node_dist = {}
        for node in nx.topological_sort(dag):
            node_w = dag.nodes[node]['duration']
            dists = [node_dist[n] for n in dag.predecessors(node)]
            if len(dists) > 0:
                node_w += max(dists)

            node_dist[node] = node_w
        return max(node_dist.values())


def convert_to_nx(times, machines, n_jobs, n_machines):
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


def load_network_fn(fn: str, sid: int):
    data = np.load(fn)
    data = data.astype(int)
    print(data.shape)
    times, machines = data[sid]
    print(times.shape)
    n_jobs, n_machines = data.shape[2:]
    return times, machines, convert_to_nx(times, machines, n_jobs, n_machines)


def draw_networks(g1, g2=None):
    with plt.style.context('ggplot'):
        # pos = nx.spring_layout(g1, seed=7)
        pos = nx.kamada_kawai_layout(g1)
        nx.draw_networkx_nodes(g1, pos, node_size=7)
        nx.draw_networkx_labels(g1, pos)
        nx.draw_networkx_edges(g1, pos, arrows=True)
        if g2:
            nx.draw_networkx_edges(g2, pos, edge_color='r')
        plt.draw()
        plt.savefig("dgraph_aco.png")
        # plt.show()


def generate_random_machines(n, m):
    machines = np.zeros((n, m), dtype=int)
    for i in range(n):
        machines[i] = np.random.permutation(m) + 1
    return machines


def ortools_api(jobs, machines):
    r""" Solve the problem in OR-Tools
    """
    import collections
    from ortools.sat.python import cp_model
    # process jobs and machines
    n_jobs, n_machines = jobs.shape
    # compute horizon dynamically as the sum of all durations
    horizon = jobs.sum()
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pos', type=int, default=0)
    args = parser.parse_args()

    times, machines, (g1, g2) = load_network_fn(args.data, args.pos)
    n, m = times.shape

    # case 1 example:
    # https://developers.google.com/optimization/scheduling/job_shop
    # n, m = 3, 3
    # times = np.array([[3, 2, 2], [2, 1, 4], [4, 3, 0]])
    # machines = np.array([[1, 2, 3], [1, 3, 2], [2, 3, 1]])

    # case2: random jobs (n=10, m=8)
    #n = 5
    #m = 4
    #times = np.random.randint(1, 10, (n, m))
    #machines = generate_random_machines(n, m)

    #g1, g2 = convert_to_nx(times, machines, n, m)
    p = DGProblem(g1, g2)
    model = ACOR.OriginalACOR(epoch=50, pop_size=10)
    # model = PSO.OriginalPSO(epoch=100, pop_size=100, seed=10)
    model.solve(p, mode="swarm")
    # print(model.g_best.solution)
    print("ACO - best solution", model.g_best.target.fitness)

    res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    draw_networks(res_dag)

    # solve the problem in ortools
    solver = ortools_api(times, machines - 1)

    aco_best = []
    for i in range(len(model.history.list_global_best)):
        aco_best.append(model.history.list_global_best[i].target.fitness)

    ortools_best = solver.objective_value

    fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    axs.plot(np.arange(len(aco_best)), aco_best, label='ACO')
    axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Makespan')
    axs.legend()
    plt.savefig(f"ACO_vs_ORTools_{n}_{m}.png")
    # plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    main()
