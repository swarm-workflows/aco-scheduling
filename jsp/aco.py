
import random

from jsp.ant import Ant
from jsp.disjunctive_graph import DisjunctiveGraph


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

        for edge in self.graph.ConjGraph.edges():
            self.graph.set_edge_pheromones(edge[0], edge[1])
            # self.graph.set_edge_pheromones(edge[0], edge[1], "disjunctive")
        # for edge in self.graph.DisjGraph.edges():
        #     self.graph.set_edge_pheromones(edge[0], edge[1], "disjunctive", 1.0)

    def find_minimum_makespan(self, source, target, num_ants, **kwargs):
        r""" Find the minimum makespan from the source to the target node in the graph

        Args:
            source (str): The source node in the graph
            target (str): The target node in the graph
            num_ants (int): The number of ants to be inited

        Returns:
            Tuple[List, float]: A tuple containing the path and the cost of the path
        """
        # send out ants to search for the destination node
        self._deploy_search_ants(source, target, num_ants)
        solution_ant = self._deploy_solution_ant(source, target)
        return solution_ant.path, solution_ant.path_cost
        # get the final solution based on the pheromone greedy approach, ie.e, for
        # the bi-directional disjunctive edges, assign direction based on the
        # pheromone level.
        #
        # after getting the DAG, calculate the final solution based on
        #  `s_v + p_v = f_v, s_v = max_{u->v}(f_u)`

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

            self._deploy_forward_search_ants()
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
        r""" Deploy forward search ants in the graph
        """
        for ant_id, ant in enumerate(self.search_ants):
            for step in range(self.ant_max_steps):
                if ant.reached_destination():
                    # print(ant.path, ant.visited_nodes)
                    print(f"Ant {ant_id}", ant.path)
                    ant.is_fit = True
                    break
                ant.take_step(step)

    def _deploy_backward_search_ants(self):
        r""" Deploy fit search ants back towards their source node while dropping pheromones on the path
        """
        for ant in self.search_ants:
            if ant.is_fit:
                ant.deposit_pheromones_on_path()
