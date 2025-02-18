import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
# For ACM fonts
matplotlib.rcParams['pdf.fonttype'] = 42


def draw_networks(dg):
    r""" Draw the disjunctive graph via networkx API.

    Args:
        dg (nx.Graph): The disjunctive graph to draw.
    """

    with plt.style.context('ggplot'):
        pos = nx.spring_layout(dg, seed=7)
        # pos = nx.kamada_kawai_layout(g1)
        # pos=0
        nx.draw_networkx_nodes(dg, pos, node_size=7)
        nx.draw_networkx_labels(dg, pos)
        nx.draw_networkx_edges(dg, pos, arrows=True)
        plt.draw()
        plt.savefig("dgraph_aco.png")
        # plt.show()
