from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Task(object):
    r""" Task representation for JSP

    * For each task, (j_id, t_id, m_id, p_time, s_time, f_time)
        * j_id: job id
        * t_id: task id
        * m_id: machine id
        * p_time: processing time
        * s_time: start time
        * f_time: finish time = s_time + p_time
    """

    def __init__(self,
                 name,
                 pos,
                 p_time=0,
                 m_id=-1,
                 scheduled=True,
                 color="tab:gray",
                 job=-1,
                 start_time=None,
                 finish_time=None):
        r""" Initialize the task

        Args:
            name (str): name of the task.
            pos (tuple): position of the task in the graph.
            p_time (int): processing time of the task.
            m_id (int): machine id of the task.
            scheduled (bool): indicates if the task is scheduled or not.
            color (str): color of the task.
            job (int): job id of the task.
            start_time (int): start time of the task.
            finish_time (int): finish time of the task.
        """
        self.name = name
        self.pos = pos
        self.p_time = p_time
        self.m_id = m_id
        self.scheduled = scheduled
        self.color = color
        self.job = job
        self.start_time = start_time
        self.finish_time = finish_time


class DisjunctiveGraph(object):
    r""" Disjunctive graph representation using `networkx` for job shop scheduling problem.

    Attributes:
        jobs (list): list of jobs where each job is a list of tasks.
        ConjGrapah (networkx.DiGraph): disjunctive graph representation of the job shop scheduling problem.
        DisjGraph (networkx.Graph): disjunctive graph representation of the job shop scheduling problem.
        n_jobs (int): number of jobs in the disjunctive graph.
        n_machines (int): number of machines in the disjunctive graph.
        max_job_length (int): maximum length of a job in the disjunctive graph.
    """

    def __init__(self, jobs=None, times=None, machines=None, c_map="gist_rainbow"):
        r""" Initialize the disjunctive graph.

        Args:
            jobs (np.ndarray): array of jobs where each job is a list of tasks.
                Each task is a list of two elements: machine id and processing time.
            times (np.ndarray): array of processing times for each job.
            machines (np.ndarray): array of machine ids for each job.
            c_map (str): colormap for the machines.
                See also: https://matplotlib.org/stable/tutorials/colors/colormaps.html

        Example:
            ```
            jobs = [[[0, 1], [1, 2], [2, 3]],
                    [[1, 2], [0, 3], [2, 0]],
                    [[2, 2], [0, 1], [1, 2]]]
            dg = DisjunctiveGraph(jobs=jobs)
            dg.draw_graph()
            ```

        Notes:
            - machine id starts from 0
            - processing time is a positive value

        """
        if jobs is not None:
            if isinstance(jobs, list):
                # convert the list into numpy
                jobs = np.array(jobs)
            elif not isinstance(jobs, np.ndarray):
                raise ValueError("Invalid input type for jobs. Expected list or numpy.ndarray.")

            self.times = jobs[:, :, 1]
            self.machines = jobs[:, :, 0]
            self.jobs = jobs
        elif times is not None and machines is not None:
            self.times = times
            self.machines = machines
            self.jobs = np.array([[(machines[i][j], times[i][j])
                                   for j in range(machines.shape[1])]
                                 for i in range(machines.shape[0])])
        else:
            raise ValueError("Invalid input type for jobs. Expected list or numpy.ndarray.")

        self.DisjGraph = nx.DiGraph()
        self.c_map = c_map
        self._process_jobs()

    @property
    def n_jobs(self):
        r""" Number of jobs in the disjunctive graph.

        Returns:
            int: number of jobs in the disjunctive graph.
        """
        return len(self.jobs)

    @property
    def n_machines(self):
        r""" Number of machines in the disjunctive graph.

        Returns:
            int: number of machines in the disjunctive graph.
        """
        return max(m for job in self.jobs for m, _ in job) + 1

    @property
    def max_job_length(self):
        r""" Maximum length of a job in the disjunctive graph.

        Returns:
            int: maximum length of a job in the disjunctive graph.
        """
        return max(len(job) for job in self.jobs)

    @property
    def total_tasks(self):
        r""" Total number of tasks in the disjunctive graph.

        Returns:
            int: total number of tasks in the disjunctive graph.
        """
        return sum(len(job) for job in self.jobs)

    @property
    def machine_tasks(self):
        r""" Tasks grouped by machine id

        Returns:
            dict: tasks grouped by machine id.
        """
        machine_tasks = defaultdict(list)
        for j_id, job in enumerate(self.jobs):
            for t_id, (m, p) in enumerate(job):
                machine_tasks[m].append((j_id, t_id))
        return machine_tasks

    def _process_jobs(self):
        r""" Process jobs to create disjunctive graph (hybrid conjunctive and disjunctive graph).

        - create source node
        - create target node
        - create task nodes
        - create conjunctive edges
          - from source to first task of each job
          - from last task of each job to target
          - from task i-1 to task i of each job
        - create disjunctive edges
          - add bidirectional disjunctive edges between tasks if tasks are on the same machine
        """

        # prepare colors for machines, select the desired colormap
        c_map = plt.cm.get_cmap(self.c_map)
        # create a list with numbers from 0 to 1 with n_machines elements
        c_machines = np.linspace(0, 1, self.n_machines, dtype=np.float32)
        # map the numbers to colors
        self._machine_colors = {m_id: c_map(val) for m_id, val in enumerate(c_machines)}

        ''' initiate the conjunctive graph '''
        # add dummy `source` node
        self.DisjGraph.add_node("S",
                                j_id=-1,
                                t_id=-1,
                                m_id=-1,
                                p_time=0,
                                start_time=0,
                                finish_time=0,
                                pos=(-1, (self.n_jobs - 1) / 2),
                                color="tab:gray",
                                )

        # create task nodes
        for j_id, job in enumerate(self.jobs):
            t_id = 0
            interval = (self.max_job_length - 1) / (len(job) - 1)
            for m, p in job:
                self.DisjGraph.add_node(f"J_{j_id}_{t_id}",
                                        j_id=j_id,
                                        t_id=t_id,
                                        m_id=m,
                                        p_time=p,
                                        start_time=None,
                                        finish_time=None,
                                        pos=(t_id * interval, j_id),
                                        color=self._machine_colors[m],
                                        )
                t_id += 1

        # add dummy `target` node
        self.DisjGraph.add_node("T",
                                j_id=-1,
                                t_id=-1,
                                p_time=0,
                                m_id=-1,
                                job=-1,
                                start_time=None,
                                finish_time=None,
                                pos=(self.max_job_length, (self.n_jobs - 1) / 2),
                                color="tab:gray",
                                )

        ''' build the conjunctive edges '''
        # add source to first task of each job
        for j_id, job in enumerate(self.jobs):
            self.DisjGraph.add_edge("S",
                                    f"J_{j_id}_0",
                                    weight=0,
                                    assigned=True,
                                    type="conjunctive")
        # add conjunctive edges from task i-1 to task i of each job
        for j_id, job in enumerate(self.jobs):
            for i in range(1, len(job)):
                self.DisjGraph.add_edge(f"J_{j_id}_{i-1}",
                                        f"J_{j_id}_{i}",
                                        weight=0,
                                        assigned=True,
                                        type="conjunctive")
        # add last task of each job to target
        for j_id, job in enumerate(self.jobs):
            self.DisjGraph.add_edge(f"J_{j_id}_{len(job)-1}",
                                    "T",
                                    weight=0,
                                    assigned=True,
                                    type="conjunctive")

        ''' build the disjunctive edges '''
        # NOTE: add bidirectional edges between tasks on the same machine
        for m, tasks in self.machine_tasks.items():
            # add pair wise edges among tasks
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    # u -> v
                    self.DisjGraph.add_edge(f"J_{tasks[i][0]}_{tasks[i][1]}",
                                            f"J_{tasks[j][0]}_{tasks[j][1]}",
                                            weight=0,
                                            assigned=False,
                                            type="disjunctive")
                    # v -> u
                    self.DisjGraph.add_edge(f"J_{tasks[j][0]}_{tasks[j][1]}",
                                            f"J_{tasks[i][0]}_{tasks[i][1]}",
                                            weight=0,
                                            assigned=False,
                                            type="disjunctive")

    def draw_graph(self, conjunctive=True, disjunctive=False, edge_status=False):
        r""" Draw the disjunctive graph

        Args:
            conjunctive (bool): draw conjunctive edges or not.
            disjunctive (bool): draw disjunctive edges or not.
            edge_status (bool): show pheromone values on the edges or not.

        Example:
            ```
            dg.draw_graph(conjunctive=True, disjunctive=True, edge_status=True)
            ```
        """

        # TODO: specify the figsize from arg, if not assigned, use the default value
        # plt.figure(figsize=(self.max_job_length, self.n_jobs / 4),
        #            tight_layout=True,
        #            dpi=300)
        pos = nx.get_node_attributes(self.DisjGraph, 'pos')
        nx.draw_networkx_nodes(
            self.DisjGraph,
            pos,
            node_color=nx.get_node_attributes(self.DisjGraph, 'color').values(),
            node_size=300)
        nx.draw_networkx_labels(
            self.DisjGraph,
            pos,
            font_size=8,
            labels={node: f"{data['m_id']}, {data['p_time']}" for node, data in self.DisjGraph.nodes(data=True)}
        )

        # draw conjunctive edges
        if conjunctive:
            conj_edges = [(u, v) for u, v, d in self.DisjGraph.edges(data=True) if d['type'] == 'conjunctive']
            nx.draw_networkx_edges(
                self.DisjGraph,
                pos,
                edgelist=conj_edges,
                edge_color='k',
                style="solid",
                width=1)

        # draw disjunctive edges
        if disjunctive:
            disj_edges = [(u, v) for u, v, d in self.DisjGraph.edges(data=True)
                                     if d['type'] == 'disjunctive']
            disj_colors = [self._machine_colors[self.DisjGraph.nodes[u]['m_id']]
                           for u, v in disj_edges]
            nx.draw_networkx_edges(
                self.DisjGraph,
                pos,
                edgelist=disj_edges,
                # edge_color='r',
                edge_color=disj_colors,
                style="dashed",
                arrows=True,
                connectionstyle="arc3,rad=0.1",
                width=1)

        if edge_status:
            # labels = {edge: attrs.get('pheromones', 1)
            #           for edge, attrs in nx.get_edge_attributes(self.ConjGraph, "pheromones").items()}

            all_edges = nx.get_edge_attributes(self.DisjGraph, "pheromones")

            # Separate edges into conjunctive and disjunctive
            conjunctive_edges = {edge: f"{attrs:.2f}" for edge, attrs in all_edges.items()
                                 if self.DisjGraph.edges[edge]['type'] == 'conjunctive'}
            disjunctive_edges = {edge: f"{attrs:.2f}" for edge, attrs in all_edges.items()
                                 if self.DisjGraph.edges[edge]['type'] == 'disjunctive'}

            # Draw labels for conjunctive edges
            nx.draw_networkx_edge_labels(self.DisjGraph,
                                         pos,
                                         edge_labels=conjunctive_edges,
                                         font_size=8)

            # Draw labels for disjunctive edges with connectionstyle
            nx.draw_networkx_edge_labels(self.DisjGraph,
                                         pos,
                                         edge_labels=disjunctive_edges,
                                         font_size=8,
                                         connectionstyle="arc3,rad=0.1")

    def set_edge_pheromones(self, u, v, type="conjunctive", pheromone_value=1e-10):
        r""" Set pheromone value on the edge (u, v)

        Args:
            u (str): source node
            v (str): destination node
            type (str): "conjunctive" or "disjunctive"
            pheromone_value (float): pheromone value to be set on the edge (u, v)
        """
        if self.DisjGraph.edges[u, v]["type"] == "conjunctive":
            self.DisjGraph[u][v]["pheromones"] = pheromone_value
        elif self.DisjGraph.edges[u, v]["type"] == "disjunctive":
            self.DisjGraph[u][v]["pheromones"] = pheromone_value
            self.DisjGraph[v][u]["pheromones"] = pheromone_value
        else:
            raise ValueError("Invalid type. Expected 'conjunctive' or 'disjunctive'.")

    def get_edge_pheromones(self, u, v):
        r""" Get pheromone value on the edge (u, v)

        Args:
            u (str): source node
            v (str): destination node

        Returns:
            float: pheromone value on the edge (u, v)
        """
        if self.DisjGraph.has_edge(u, v):
            return self.DisjGraph[u][v]["pheromones"]
        elif self.DisjGraph.has_edge(u, v):
            return self.DisjGraph[u][v]["pheromones"]
        else:
            raise ValueError(f"No edge between {u} and {v} in either graph.")

    def get_edge_cost(self, u, v):
        r""" Get cost of the edge (u, v), the cost is the processing time needed for node v.

        Args:
            u (str): source node
            v (str): destination node

        Notes:
            - Cost of the edge (u, v) can be an extended version of disjunctive graph for resilient job shop scheduling
              problem, including but not limited to: networking, delay, processing time, etc.
            - Add edge processing time to the node.
        """
        # if (u, v) in self.DisjGraph.edges():
        #     return self.DisjGraph.edges[(u, v)]['p_time']
        # else:
        #     raise ValueError(f"No edge between {u} and {v} in the graph.")
        return self.DisjGraph.nodes[v]['p_time']

    def get_neighbors(self, node):
        r""" Get candidates neighbors from node,
          including the successor in conjunctive nodes and neighbors in disjunctive nodes

        Args:
            node (str): node name

        Returns:
            list: list of neighbors of the node
        """
        return list(self.DisjGraph.neighbors(node))

    def extract_subgraph_with_hop(self, node, hop=2):
        r"""Extract a subgraph with a given hop distance from a node.

        Args:
            node (str): The node from which to start.
            hop_distance (int): The hop distance.

        Returns:
            nx.DiGraph: The extracted subgraph.
        """
        # Use ego_graph to get the subgraph with the specified hop distance
        subgraph = nx.ego_graph(self.DisjGraph, node, radius=hop, undirected=False)
        return subgraph
