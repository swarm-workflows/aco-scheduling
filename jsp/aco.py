
import random

from .ant import Ant
from .disjunctive_graph import DisjunctiveGraph
from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class ACO(object):

    def __init__(self,
                 graph=None,
                 ant_max_steps=100,
                 num_iterations=100,
                 ant_random_init=False,
                 evaporation_rate=0.5,
                 alpha=0.7,
                 beta=0.3):

        self.graph = graph
        self.ant_max_steps = ant_max_steps
        self.num_iterations = num_iterations
        self.ant_random_init = ant_random_init
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.search_ants = []
        self.best_sol = None
        self.best_makespan = np.inf

        for edge in self.graph.ConjGraph.edges():
            self.graph.set_edge_pheromones(edge[0], edge[1])
            # self.graph.set_edge_pheromones(edge[0], edge[1], "disjunctive")
        # for edge in self.graph.DisjGraph.edges():
        #     self.graph.set_edge_pheromones(edge[0], edge[1], "disjunctive", 1.0)

    def calculate_makespan(self, dag, **kwargs):
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


        if not nx.is_directed_acyclic_graph(dag):
            return None, np.inf
 
        #algorithm to calculate makespan with respect to precedence and machine constraints
        node_dist = {}
        for node in nx.topological_sort(dag):
            node_w = dag.nodes[node]['p_time']
            dists = [node_dist[n] for n in dag.predecessors(node)]
            if len(dists) > 0:
                node_w += max(dists)

            node_dist[node] = node_w

        makespan = max(node_dist.values())
        
        return makespan

    def build_dag(self, graph):
        """ Create a directed acyclic graph so that makespan of schedule can be calculated.
        """

        dag = graph.ConjGraph.copy()
        adjList = deepcopy(dag._adj)

        edges = deepcopy(dag.edges)

        #Remove disjunctive edges from the graph
        for idx, _ in enumerate(edges):
            start, end = _
            if adjList[start][end]['type'] == 'disjunctive':
                dag.remove_edge(start, end)

        assert isinstance(dag, nx.DiGraph)

        #Add disjunctive edges back to the graph
        for idx, node in enumerate(adjList): #source and target nodes have no disjunctive edges
            if node == 'S' or node == 'T':
                continue
            
            disjunctive_edges = dict(node for node in adjList[node].items() if node[1]['type'] == 'disjunctive')

            #For disjunctives edges (u->v) and (v->u), only add disj edge with higher pheromone value
            for task in disjunctive_edges:

                edge_pheromone = adjList[node][task]['pheromones']
                reverse_edge_pheromone = adjList[task][node]['pheromones']
                
                if edge_pheromone > reverse_edge_pheromone:
                    dag.add_edge(node, task)
                    if nx.is_directed_acyclic_graph(dag):
                        continue
                    else:
                        dag.remove_edge(node, task)
                else:
                    dag.add_edge(task, node)
                    if nx.is_directed_acyclic_graph(dag):
                        continue
                    else:
                        dag.remove_edge(task, node)
        return dag
    
    def draw_networks(self, g1, g2=None):
        with plt.style.context('ggplot'):
            pos = nx.spring_layout(g1, seed=7)
            #pos = nx.kamada_kawai_layout(g1)
            #pos=0
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
        for _ in range(self.num_iterations):
            self.search_ants = []

            for iter in range(num_ants):
                init_point = random.choice(list(self.graph.ConjGraph.nodes)) if self.ant_random_init else source

                ant = Ant(self.graph,
                          init_point,
                          target,
                          visited_nodes=set(),
                          path=[],
                          alpha=self.alpha,
                          beta=self.beta)
                self.search_ants.append(ant)

            #Send num_ants out to explore the graph
            self._deploy_forward_search_ants()
            #Send ants that reached target backwards to deposit pheromones
            self._deploy_backward_search_ants()

            #Find solutions on each iteration of ants crawling
            curr_sol = self.build_dag(self.graph)
            curr_makespan = self.calculate_makespan(curr_sol)

            #Keep best solution
            if curr_makespan < self.best_makespan:
                self.best_sol = curr_sol
                self.best_makespan = curr_makespan

        return curr_sol

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
        r""" Deploy forward search ants in the graph
        """
        for ant_id, ant in enumerate(self.search_ants):
            for step in range(self.ant_max_steps):
                if ant.reached_destination():
                    # print(ant.path, ant.visited_nodes)
                    print(f"Ant {ant_id}", ant.path)
                    #ant_graph = self.build_dag(ant.graph)
                    #ant.makespan = self.calculate_makespan(ant_graph)
                    #print(f"Ant {ant_id} {ant.path} {ant.makespan}")
                    ant.is_fit = True
                    break
                ant.take_step(step)

    def _deploy_backward_search_ants(self):
        r""" Deploy fit search ants back towards their source node while dropping pheromones on the path
        """
        for ant in self.search_ants:
            if ant.is_fit:
                ant.deposit_pheromones_on_path()
