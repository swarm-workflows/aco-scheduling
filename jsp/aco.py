import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ortools.sat.python import cp_model
from collections import defaultdict, namedtuple


from .ant import Ant
from .disjunctive_graph import DisjunctiveGraph


class ACO(object):
    r""" Ant colony optimization for job-shop scheduling problem """

    def __init__(self,
                 graph=None,
                 ant_max_steps=100,
                 num_iterations=100,
                 ant_random_init=True,
                 rho=0.9,
                 alpha=0.2,
                 beta=0.8,
                 tau_min=0.1,
                 tau_max=10.0,
                 ):

        self.graph = graph
        self.ant_max_steps = ant_max_steps
        self.num_iterations = num_iterations
        self.ant_random_init = ant_random_init
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.search_ants = []

        # initialize pheromones on the graph, including both conj and disj edges
        for edge in self.graph.DisjGraph.edges():
            self.graph.set_edge_pheromones(edge[0], edge[1])

    def find_minimum_makespan(self, source, target, num_ants, **kwargs):
        r""" Find the minimum makespan from the source to the target node in the graph

        Args:
            source (str): The source node in the graph
            target (str): The target node in the graph
            num_ants (int): The number of ants to be inited

        Returns:
            Tuple[List, float]: A tuple containing the path and the cost of the path
        """

        # get the final solution based on the pheromone greedy approach, ie.e, for
        # the bi-directional disjunctive edges, assign direction based on the
        # pheromone level.
        #
        # after getting the DAG, calculate the final solution based on
        #  `s_v + p_v = f_v, s_v = max_{u->v}(f_u)`

        # TODO: track unvisited nodes
        self.unexplored_nodes = set(self.graph.DisjGraph.nodes)
        # send out ants to search for the destination node
        self._deploy_search_ants(source, target, num_ants)

        # retrieve the solution ant
        solution_ant = self._deploy_solution_ant(source, target)

        self.sol_dag = self.build_dag()

        # print(self.sol_dag)
        if not nx.is_directed_acyclic_graph(self.sol_dag):
            return None, np.inf

        # algorithm to calculate makespan with respect to precedence and machine constraints
        # node_dist = {}
        # for node in nx.topological_sort(self.sol_dag):
        #     node_w = self.sol_dag.nodes[node]['p_time']
        #     dists = [node_dist[n] for n in self.sol_dag.predecessors(node)]
        #     if len(dists) > 0:
        #         node_w += max(dists)

        #     node_dist[node] = node_w

        # makespan = max(node_dist.values())
        makespan = self.calulate_makespan(self.sol_dag)

        return solution_ant.path, makespan

    def calulate_makespan(self, dag):
        r""" Calculate the makespan of the schedule.

        Args:
            dag (nx.DiGraph): The directed acyclic graph representing the schedule.
        """
        assert nx.is_directed_acyclic_graph(dag)
        # algorithm to calculate makespan with respect to precedence and machine constraints
        node_dist = {}
        for node in nx.topological_sort(dag):
            node_w = dag.nodes[node]['p_time']
            dists = [node_dist[n] for n in dag.predecessors(node)]
            if len(dists) > 0:
                node_w += max(dists)

            node_dist[node] = node_w

        makespan = max(node_dist.values())
        return makespan

    def build_dag(self):
        r""" Create a directed acyclic graph so that makespan of schedule can be calculated. """

        dag = self.graph.DisjGraph.copy()
        adjList = deepcopy(dag._adj)
        edges = deepcopy(dag.edges)

        disj_edges = [edge for edge in edges if dag._adj[edge[0]][edge[1]]['type'] == 'disjunctive']
        # for m_id, tasks in self.graph.machine_tasks.items():
        #     for i in range(len(tasks)):
        #         for j in range(i + 1, len(tasks)):
        #             u = f"J_{tasks[i][0]}_{tasks[i][1]}"
        #             v = f"J_{tasks[j][0]}_{tasks[j][1]}"
        #             if self.graph.DisjGraph[u][v]["pheromones"] >= self.graph.DisjGraph[v][u]["pheromones"]:
        #                 self.graph.DisjGraph.remove_edge(v, u)
        #             else:
        #                 self.graph.DisjGraph.remove_edge(u, v)

        # Remove disjunctive edges from the graph
        for idx, _ in enumerate(edges):
            start, end = _
            if dag._adj[start][end]['type'] == 'disjunctive':
                dag.remove_edge(start, end)

        assert isinstance(dag, nx.DiGraph)

        # Add disjunctive edges back to the graph
        for idx, node in enumerate(adjList):  # source and target nodes have no disjunctive edges
            if node == 'S' or node == 'T':
                continue

            disjunctive_edges = dict(node for node in adjList[node].items() if node[1]['type'] == 'disjunctive')

            # For disjunctives edges (u->v) and (v->u), only add disj edge with higher pheromone value
            for task in disjunctive_edges:

                edge_pheromone = adjList[node][task]['pheromones']
                reverse_edge_pheromone = adjList[task][node]['pheromones']

                if edge_pheromone > reverse_edge_pheromone:
                    dag.add_edge(node, task,
                                 type='disjunctive',
                                 pheromones=edge_pheromone,
                                 assigned=True)

                    if nx.is_directed_acyclic_graph(dag):
                        continue
                    else:
                        dag.remove_edge(node, task)
                else:
                    dag.add_edge(task, node,
                                 type='disjunctive',
                                 pheromones=reverse_edge_pheromone,
                                 assigned=True)
                    if nx.is_directed_acyclic_graph(dag):
                        continue
                    else:
                        dag.remove_edge(task, node)
        self.graph.DisjGraph = dag
        return dag

    def draw_networks(self, g1, g2=None):
        r""" Draw the networkx graph

        Args:
            g1 (nx.Graph): The networkx graph to be drawn
            g2 (nx.Graph): The networkx graph to be drawn
        """
        with plt.style.context('ggplot'):
            pos = nx.spring_layout(g1, seed=7)
            # pos = nx.kamada_kawai_layout(g1)
            # pos=0
            nx.draw_networkx_nodes(g1, pos, node_size=7)
            nx.draw_networkx_labels(g1, pos)
            nx.draw_networkx_edges(g1, pos, arrows=True)
            if g2:
                nx.draw_networkx_edges(g2, pos, edge_color='r')
            plt.draw()
            plt.savefig("dgraph_aco.png")
            # plt.show()

    def _deploy_search_ants(self, source, target, num_ants, **kwargs):
        r"""Deploy search ants that traverse the graph to find the shortest path

        Args:
            source(str): The source node in the graph
            destination(str): The destination node in the graph
            num_ants(int): The number of ants to be inited
        """
        for iter in range(self.num_iterations):
            self.search_ants = []
            # print(f"Iter {iter}:", end=" ")

            for ant_idx in range(num_ants):
                # deploy ant randomly on disjunctive graph
                init_point = random.choice(list(self.graph.DisjGraph.nodes)) if self.ant_random_init else source

                ant = Ant(self.graph,
                          init_point,
                          target,
                          visited_nodes=set(),
                          path=[],
                          alpha=self.alpha,
                          beta=self.beta)
                self.search_ants.append(ant)

            # move ants in the graph
            self._deploy_forward_search_ants()
            # update pheromones on the graph
            self._deploy_backward_search_ants()

    def _deploy_solution_ant(self, source, target):
        r"""Deploy the pheromone-greedy solution to find minimum makespan

        Args:
            source (str): The source node in the graph
            destination (str): The destination node in the graph

        Returns:
            Ant: The solution ant with the computed shortest path and cost
        """
        ant = Ant(self.graph, source, target, visited_nodes=set(), path=[], is_solution_ant=True)
        step = 0
        while not ant.reached_destination():
            ant.take_step(step)
            step += 1
            # print(step, ant.path)
        return ant

    def _deploy_forward_search_ants(self):
        r""" Deploy forward search ants in the graph.

        Notes:
          - For each ant, it moves to the next node in the graph
            until the ant reaches the destination node or the maximum number of steps is reached
        """
        for ant_id, ant in enumerate(self.search_ants):
            for step in range(self.ant_max_steps):
                if ant.reached_destination():
                    # print(ant.path, ant.visited_nodes)
                    # print(f"Ant {ant_id}", ant.path)
                    ant.is_fit = True
                    break
                else:
                    ant.take_step(step)

    def _deploy_backward_search_ants(self):
        r""" Deploy fit search ants back towards their source node while dropping pheromones on the path

        Notes:
          - For each ant reaches its destination, drop pheromones on the path.
        """
        for ant in self.search_ants:
            # if ant.is_fit:
            #     ant.deposit_pheromones_on_path()
            ant.deposit_pheromones_on_path()


class ACO_LS(ACO):
    r""" Ant colony optimization with local search for job-shop scheduling problem """

    def __init__(self,
                 graph=None,
                 ant_max_steps=100,
                 num_iterations=100,
                 ant_random_init=True,
                 rho=0.5,
                 alpha=0.7,
                 beta=0.3,
                 n_subgraphs=10,):

        super().__init__(graph,
                         ant_max_steps,
                         num_iterations,
                         ant_random_init,
                         rho,
                         alpha,
                         beta)
        self.n_subgraphs = n_subgraphs

    def find_minimum_makespan(self, source, target, num_ants, **kwargs):
        r""" Find the minimum makespan from the source to the target node in the graph

        Args:
            source (str): The source node in the graph
            target (str): The target node in the graph
            num_ants (int): The number of ants to be inited

        Returns:
            Tuple[List, float]: A tuple containing the path and the cost of the path
        """

        # get the final solution based on the pheromone greedy approach, ie.e, for
        # the bi-directional disjunctive edges, assign direction based on the
        # pheromone level.
        #
        # after getting the DAG, calculate the final solution based on
        #  `s_v + p_v = f_v, s_v = max_{u->v}(f_u)`

        # send out ants to search for the destination node
        self._deploy_search_ants(source, target, num_ants)

        # retrieve the solution ant
        solution_ant = self._deploy_solution_ant(source, target)

        self.sol_dag = self.build_dag()

        # print(self.sol_dag)
        if not nx.is_directed_acyclic_graph(self.sol_dag):
            return None, np.inf

        makespan = self.calulate_makespan(self.sol_dag)
        self.local_search()
        return solution_ant.path, makespan

    def local_search(self, n_hops=2, n_samples=5):
        r""" Perform local search on the graph
        Notes:
            1. random pick a node in graph, and sample an egograph with n_hops
            2. solve the subgraph using OR-Tools
            3. update the direction of disjunctive edges based on the solution
            4. repeat the process until no improvement is made

        Args:
            hop (int): The radius of the ego graph
            n_samples (int): The number of subgraphs to be taken
        """
        subgraph_solutions = []
        for _ in range(n_samples):
            node = random.choice(list(self.graph.DisjGraph.nodes))
            # TODO: the ego graph should be a subgraph of original disjunctive graph
            subG = nx.ego_graph(self.graph.DisjGraph, node, radius=n_hops)

            # TODO: update the direction of disjunctive edges based on the solution
            solution = self.solve_subgraph_with_ortools(subG)
            if solution:
                # Update the direction of disjunctive edges based on the solution
                for u, v in subG.edges:
                    if solution[u] < solution[v]:
                        self.graph.DisjGraph[u][v]['direction'] = 'forward'
                    else:
                        self.graph.DisjGraph[u][v]['direction'] = 'backward'

                new_makespan = self.calculate_makespan()
                if new_makespan < current_makespan:
                    current_makespan = new_makespan
                    improvement = True
                else:
                    improvement = False
                subgraph_solutions.append(solution)

        self.update_pheromones(subgraph_solutions)

    def update_pheromones(self, subgraph_solutions, gamma=1):
        r""" Update pheromones after the results of local search solutions

        .. math::
            \tau_{ij} = \tau_{ij} + \gamma prob_{ij}, where
            - \gamma is the factor of contribution from local search
            - prob_{ij} is the probability of direction from the solution of subgraphs, prob_{ij} + prob_{ji} = 1

        Args:
            subgraph_solutions (List[Dict]): The solutions of subgraphs
            gamma (float): The factor of contribution from local search
        """
        # Reset pheromones
        counts = {}
        for u, v in self.graph.DisjGraph.edges():
            counts[(u, v)] = 0

        # Count the times of direction from solution of subgraphs
        for subgraph_index, solutions in enumerate(subgraph_solutions):
            direction_count = {}
            total_count = 0
            for solution in solutions:
                for u, v in solution:
                    direction_count[(u, v)] += 1
                    total_count += 1

            # Normalize the pheromones so that they sum to 1
            for direction in direction_count:
                direction_count[direction] /= total_count

            # Update pheromones based on normalized direction count
            self.graph.DisjGraph.edges[(u, v)] += gamma * direction_count

    def solve_subgraph_with_ortools(self, subgraph):
        r""" Solve the subgraph using OR-Tools

        Notes:
            - OR-Tools is used to solve the subgraph
            - The input of subgraph is not necessary to be a matrix form, but list of tuples.

        Args:
            subgraph (nx.Graph): The subgraph to be solved

        Returns:
            Dict: The solution to the subgraph
        """
        # TODO: revise and reuse the utils function `ortools_api` to solve the subgraph
        # Create the model
        model = cp_model.CpModel()

        # From subgraph, extract the nodes and prepare data as job_data = [[(machine_id, duration), ...], ...]
        # TODO: build a job_data array from subgraph, where each element is a list of tuples (machine_id, duration)
        job_data = defaultdict(list)
        for node in subgraph.nodes:
            if node in ['S', 'T']:
                continue
            job_data[node.split("_")[1]].append((subgraph.nodes[node]['m_id'], subgraph.nodes[node]['p_time']))
        job_data_arr = []
        for k in job_data:
            job_data_arr.append(job_data[k])

        # NOTE: solve in OR-Tools
        machines_count = 1 + max(task[0] for job in job_data_arr for task in job)

        all_machines = range(machines_count)
        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in job_data_arr for task in job)

        # Create the model.
        model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        task_type = namedtuple("task_type", "start end interval")
        # Named tuple to manipulate solution information.
        assigned_task_type = namedtuple(
            "assigned_task_type", "start job index duration"
        )

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = defaultdict(list)

        for job_id, job in enumerate(job_data_arr):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                start_var = model.new_int_var(0, horizon, "start" + suffix)
                end_var = model.new_int_var(0, horizon, "end" + suffix)
                interval_var = model.new_interval_var(
                    start_var, duration, end_var, "interval" + suffix
                )
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.add_no_overlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(job_data_arr):
            for task_id in range(len(job) - 1):
                model.add(
                    all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
                )

        # Makespan objective.
        obj_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(
            obj_var,
            [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(job_data_arr)],
        )
        model.minimize(obj_var)

        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        status = solver.solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution:")
            # Create one list of assigned tasks per machine.
            assigned_jobs = defaultdict(list)
            for job_id, job in enumerate(job_data_arr):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1],
                        )
                    )

            # Create per machine output lines.
            output = ""
            for machine in all_machines:
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
