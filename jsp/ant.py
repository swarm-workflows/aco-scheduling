""" Ant class for the ant colony optimization algorithm with disjunctive graph representation of the job shop scheduling problem.
"""
import random
import numpy as np


class Ant(object):

    def __init__(self, graph, source, target,
                 alpha=0.7,
                 beta=0.3,
                 evaporation_rate=0.1,
                 visited_nodes=set(),
                 path=[],
                 past_cost=0.0,
                 is_fit=False,
                 is_solution_ant=False):
        r""" Initialize the ant

        Args:
            graph (DisjunctiveGraph): disjunctive graph representation of the job shop scheduling problem.
            source (str): source node in the graph.
            target (str): target node in the graph.
            alpha (float): pheromone bias.
            beta (float): edge cost bias.
            visited_nodes (set): set of nodes that have been visited by the ant.
            path (list): path taken by the ant so far.
            past_cost (float): cost of the path taken by the ant so far.
            is_fit (bool): indicates if the ant has reached the destination (fit) or not (unfit).
            is_solution_ant (bool): indicates if the ant is the pheromone-greedy solution ant.
        """
        self.graph = graph
        self.source = source
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.visited_nodes = visited_nodes
        self.path = path
        self.path_cost = past_cost
        self.is_fit = is_fit
        self.is_solution_ant = is_solution_ant
        self.current_node = self.source
        self.path.append(self.source)

    def deposit_pheromones_on_path(self):
        r""" Deposit pheromones on the path taken by the ant

        .. math::
            \Delta\tau_{u, v} = \frac{1}{L_k}
        """
        # deposit pheromones on the path taken by the ant from [i, i+1]
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            # path_cost= 0
            new_pheromone_val = 1 / self.path_cost if self.path_cost != 0 else 0
            self.deposit_pheromones(u, v, new_pheromone_val)

    def deposit_pheromones(self, u, v, pheromone_amount):
        r""" Deposit pheromone on the edge (u, v)

        .. math::
            \tau_{u, v}(t+1) = (1-\rho) \tau_{u, v}(t) + \Delta\tau_{u, v}(t)`

        Args:
            u (str): source node
            v (str): destination node
            pheromone_amount (float): pheromone amount to be deposited on the edge (u, v)

            NOTE: Soon to be replaced by max_min_update_pheromones
        """
        if self.graph.DisjGraph.has_edge(u, v):
            self.graph.DisjGraph[u][v]["pheromones"] = (1 - self.evaporation_rate) * \
                self.graph.DisjGraph[u][v]["pheromones"] + pheromone_amount
        else:
            raise ValueError(f"No edge between {u} and {v} in either graph.")

    def max_min_update_pheromones(self, best_solution):
        '''TODO: Implement max_min pheromone values properly.
        Need 'best_cost' aka best makespan. Find 'best_cost' by creating a dag and calculating makespan after 1 iteration of 100 ants.
        '''
        self.pheromone *= (1 - self.rho)
        for machine, tasks in best_solution.items():
            for job, operation, start_time, finish_time in tasks:
                self.pheromone[job][machine] += 1.0 / self.best_cost

        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def reached_destination(self):
        r""" Returns if the ant has reached the destination node in the graph

        Returns:
            (bool): returns `True` if the ant has reached the destination
        """
        return self.current_node == self.target

    def take_step(self, step):
        r""" Compute and update the ant position

        Args:
            step (int): step number

        Notes:
          - The ant moves to the next node in the graph based on the probabilities of the unvisited neighbors
          - The ant updates the path and path cost
          - DAG constraints are checked when choosing the next node
        """
        # print(self.path, end="\t")
        if self.reached_destination():
            return None
        # mark current node as visited
        self.visited_nodes.add(self.current_node)
        # pick the next node for the ant
        next_node = self._choose_next_node()
        # check if ant is stuck at current node
        if not next_node:
            return

        # update path
        self.path.append(next_node)
        # update path cost
        self.path_cost += self.graph.DisjGraph.nodes[self.current_node]["p_time"]
        # update start time of next node
        # # TODO: Double check, update the s_time of the node, take min of current
        # NOTE: no need to update the s_time
        # self.graph.DisjGraph.nodes[next_node]["s_time"] = min(self.graph.DisjGraph.nodes[next_node].get("s_time", 0),
        #                                                       self.path_cost)
        # print(f"ant {self.current_node} -> {next_node}")
        # update current node
        self.current_node = next_node

    def _choose_next_node(self):
        r""" Choose the next node based on the probabilities of the unvisited neighbors
        """
        if self.reached_destination():
            return None
        unvisited_neighbors = self._get_unvisited_neighbors()

        # check if ant has no possible nodes to move to
        if len(unvisited_neighbors) == 0:
            return None

        if self.is_solution_ant:
            if len(unvisited_neighbors) == 0:
                raise Exception(
                    f"No path found from {self.source} to {self.target}"
                )

            # sort the neighbors based on the pheromone values
            candidate_node = sorted(
                unvisited_neighbors,
                key=lambda neighbor: self.graph.get_edge_pheromones(
                    self.current_node, neighbor
                ),
                reverse=True
            )
            # get the next node which is not visited
            for node in candidate_node:
                if node not in self.visited_nodes:
                    return node
            return

            # for node in candidate_node:
            #     # if add (current, node) to DisjGraph result cycles, then skip
            #     if not self.graph.creates_cycle(self.current_node, node):
            #         return node
            #         # self.graph.DisjGraph.add_edge(self.current_node, node)
            #         # break

        else:
            prob = self._calculate_edge_probabilities(unvisited_neighbors)
            # print(self.current_node, prob)
            return random.choices(list(prob.keys()), weights=list(prob.values()), k=1)[0]
            # return self._roulette_wheel_selection(prob)

    def _get_unvisited_neighbors(self):
        r"""Get unvisited neighbors of the current node

        Returns:
            (list): list of unvisited neighbors of the current node
        """
        # get the neighbors of current node
        neighbors = self.graph.get_neighbors(self.current_node)
        # exclude the node existed in visited_nodes
        unvisited_neighbors = [node for node in neighbors if node not in self.visited_nodes]
        # exclude the node existed in path
        unvisited_neighbors = [node for node in unvisited_neighbors if node not in self.path]
        return unvisited_neighbors

    def _calculate_edge_probabilities(self, unvisited_neighbors):
        r""" Calculate the probabilities of the unvisited neighbors of the current node

        .. math::
            heuristic = 1 / edge_cost
            prob = pheromone ** alpha * heuristic ** beta
            prob = prob / sum(prob)

            prob_{u, v}(t) = \frac{d_{u, v}}{\sum_{s \in N(u)} d_{u, s}}

        Args:
            unvisited_neighbors (list): list of unvisited neighbors of the current node

        Returns:
            Dict[str, float]: dictionary of probabilities of the unvisited neighbors of the current node
        """
        probabilities = {}
        all_edges_desirability = 0.0

        # calculate disireability per each edge
        # TODO: improve the implementation more efficiently -> vectorize
        for neighbor in unvisited_neighbors:
            # only calculate the node unvisited
            if neighbor not in self.path:
                edge_pheromones = self.graph.get_edge_pheromones(self.current_node, neighbor)
                edge_cost = self.graph.get_edge_cost(self.current_node, neighbor)

                current_edge_desirability = self.compute_edge_desireability(
                    edge_pheromones, edge_cost
                )
                all_edges_desirability += current_edge_desirability
                probabilities[neighbor] = current_edge_desirability
            else:
                continue

        # if only T is unvisited, set prob to be 1
        if all_edges_desirability == 0:
            for neighbor in unvisited_neighbors:
                probabilities[neighbor] = 1
            return probabilities

        # normalize the prob
        for neighbor in unvisited_neighbors:
            probabilities[neighbor] /= all_edges_desirability

        return probabilities

    def compute_edge_desireability(self, edge_pheromones, edge_cost):
        r""" Compute the desireability of the edge (u, v).

        .. math::
            d = \tau_{u, v}^{\alpha} \eta_{u, v}^{\beta}

            where
                - :math:`\tau_{u, v}` is the pheromone value of the edge (u, v)
                - :math:`\eta_{u, v}` is the inverse of the edge cost
                - :math:`\alpha` is the pheromone weight
                - :math:`\beta` is the edge cost weight

        Args:
            edge_pheromones (float): pheromone value on the edge (u, v)
            edge_cost (float): cost of the edge (u, v)

        Returns:
            float: desireability of the edge (u, v)
        """
        # edge_cost == 0, the final task to T
        # TODO: double check
        if edge_cost == 0:
            return 0
            # return edge_pheromones ** self.alpha * (1 / 1e-10) ** self.beta
        else:
            return edge_pheromones ** self.alpha * (1 / edge_cost) ** self.beta

    def _roulette_wheel_selection(self, probabilities):
        r""" Roulette wheel selection of the next node based on the probabilities of the unvisited neighbors

        Args:
            probabilities (Dict[str, float]): dictionary of probabilities of the unvisited neighbors of the current node

        Returns:
            str: the next node to be visited by the ant

        See Also:L
            * fitness proportionate selection: https://en.wikipedia.org/wiki/Fitness_proportionate_selection
        """
        sorted_probabilities = {
            k: v for k, v in sorted(probabilities.items(), key=lambda item: -item[1])
        }

        pick = random.random()
        current = 0.0
        for node, fitness in sorted_probabilities.items():
            current += fitness
            if current > pick:
                return node
        raise Exception("Edge case for roulette wheel selection")
