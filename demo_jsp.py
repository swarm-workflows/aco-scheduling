# %%
%load_ext autoreload
%autoreload 2
from jsp.disjunctive_graph import DisjunctiveGraph
from jsp.aco import ACO
import networkx as nx
import numpy as np
# %%
#
jobs = [
    [[0, 3], [1, 2], [2, 2]],
    [[0, 2], [2, 1], [1, 4]],
    [[1, 4], [2, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [5, 4], [4, 3]],
    # [[1, 2], [2, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [5, 4], [4, 3]],
    # [[1, 2], [2, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [5, 4], [4, 3]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [1, 4], [3, 3]],
    # [[0, 1], [5, 4], [4, 3], [6, 2]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
    # [[3, 2], [2, 3]],
]

jobs = np.array([[[0, 3], [1, 2], [2, 2]],
                [[0, 2], [2, 1], [1, 4]],
                [[1, 4], [2, 3], [0, 0]]])
machines = jobs[:, :, 0]
times = jobs[:, :, 1]
dg = DisjunctiveGraph(jobs=jobs, c_map="gist_rainbow")
# %%
dg.draw_graph(disjunctive=True, edge_status=False)

# %%
aco = ACO(dg, 100, num_iterations=1, ant_random_init=False)
aco_path, aco_cost = aco.find_minimum_makespan("S", "T", num_ants=4)
print(aco_path, aco_cost)
dg.draw_graph(disjunctive=True, edge_status=True)
# %%


from ortools_api import ortools_api
ortools_api(times, machines)
# %%
