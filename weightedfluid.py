"""Asynchronous Fluid Communities algorithm for community detection."""

from collections import Counter, OrderedDict
from networkx.exception import NetworkXError
from networkx.algorithms.components import is_connected
from networkx.utils import groups
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import networkx as nx
import cdlib
from cdlib import evaluation, algorithms, viz
import matplotlib.pyplot as plt
from matplotlib import cm
import pydot
import numpy

import os
import sys
import random

__all__ = ["asyn_fluidc"]


@py_random_state(3)
@not_implemented_for("directed", "multigraph")
def weight_fluid(G, k, max_iter=100, seed=None):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. Its initialization is
    random, so found communities may vary on different executions.

    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbours. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is a modified version of the algorithm described in [1]_.
    It has edge weights as the determinant given equal densities.

    Parameters
    ----------
    G : Graph

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 100.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Par??s F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
    # Initial checks
    if not isinstance(k, int):
        raise NetworkXError("k must be an integer.")
    if not k > 0:
        raise NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise NetworkXError("Fluid Communities require connected Graphs.")
    if len(G) < k:
        raise NetworkXError("k cannot be bigger than the number of nodes.")
    # Initialization
    max_density = 1.0
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {n: i for i, n in enumerate(vertices[:k])}
    for key, value in communities.items():
        G.nodes[key]["community"] = value
    density = {}
    com_to_numvertices = {}
    for vertex in communities.keys():
        com_to_numvertices[communities[vertex]] = 1
        G.nodes[vertex]["density"]=1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Loop over all vertices in graph in a random order
        vertices = list(G)
        seed.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            weights = {}
            weight_rev = {}
            neighbours = []
            for v in G[vertex]:
                neighbours.append(v)
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                    weights[v] = G[vertex][v]["weight"]
                    
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                if weights:
                    max_weight = max(weights.values())
                weight_rev = {value: key for key, value in weights.items()}
                # best_neighbour = weight_rev[max_weight]
                # best_communities = []
                # best_communities.append(communities[best_neighbour])
                best_communities = [
                    com
                    for com, freq in com_counter.items()
                    if (max_freq - freq) < 0.0001
                ]

                #Core change made to use weight as determinant from list of best_neighbours
                best_neighbours = {}
                for key, value in communities.items():
                    if value in best_communities and key in neighbours:
                        best_neighbours[key] = weights[key]

                best_neighbours_rev = {value: key for key, value in best_neighbours.items()}
                if best_neighbours_rev:
                    bestest = max(best_neighbours_rev.values())
                # for vert in best_communities:

                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Changed to pick the node with the highest edge weight
                    new_com = communities[bestest]
                    # Update previous community status
                    try:
                        if com_to_numvertices[communities[vertex]] > 1:
                            com_to_numvertices[communities[vertex]] -= 1
                            density[communities[vertex]] = (
                                max_density / com_to_numvertices[communities[vertex]]
                            )
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = (
                        max_density / com_to_numvertices[communities[vertex]]
                    )
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            break
    # Return results by grouping communities as list of vertices
    return iter(groups(communities).values())


# if __name__ == '__main__':

    # name3 = "/home/james/4F90/sg_infectious_graphs/weightededgesX_2009_05_06.out"
    # name2 = "/home/james/4F90/sg_infectious_graphs/weightededgesX_2009_06_02.out"
    # name = "/home/james/4F90/sg_infectious_graphs/weightededgesX_2009_07_15.out"
    # fh2 = open(name, "rb")
    # fh3 = open(name3, "rb")
    # testg = nx.read_weighted_edgelist(fh3)
    # my_graph2 = nx.read_weighted_edgelist(fh2)
    # fh2.close()
    # fh3.close()

    # fluid = nx.algorithms.community.asyn_fluidc(testg, 10, seed=5)
    # fluid2 = [list(x) for x in fluid]
    # fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")
    # pos = nx.spring_layout(testg, weight='weight',seed=5)
    # pos = nx.nx_pydot.graphviz_layout(testg)
    # pos2 = nx.kamada_kawai_layout(testg, weight='weight')
    # pos3 = nx.random_layout(testg)
    # louvain = algorithms.louvain(testg, weight='weight', resolution=1.5)

#     wcom = weight_fluid(testg, 9, seed=5)
#     wcoms = [list(x) for x in wcom]
#     wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")


# print(evaluation.newman_girvan_modularity(testg, wcoms2).score)
# print(evaluation.newman_girvan_modularity(testg, fluid3).score)
# print(evaluation.newman_girvan_modularity(testg, louvain).score)

#     labels = nx.get_edge_attributes(testg, 'weight')
#     # node_labels = nx.get_node_attributes(testg, name)

#     viz.plot_network_clusters(testg, wcoms2, pos,figsize=(20,20),node_size=600,cmap='gist_rainbow', plot_labels=False)
#     nx.draw_networkx_edge_labels(testg,pos, edge_labels=labels, font_size=6)
#     nx.draw_networkx_labels(testg, pos, font_size=8)
#     # viz.plot_community_graph(testg,louvain)
#     plt.savefig("Algo1_10com.png")
#     plt.show()
#     plt.close()
# ########################
#     viz.plot_network_clusters(testg, fluid3, pos,figsize=(20,20),node_size=500,cmap='gist_rainbow', plot_labels=False)
#     nx.draw_networkx_edge_labels(testg,pos, edge_labels=labels, font_size=6)
#     nx.draw_networkx_labels(testg, pos, font_size=8)
#     # viz.plot_community_graph(testg,louvain)
#     plt.savefig("Fluid_10com.png")
#     plt.show()
#     plt.close()

#     ####################################
#     viz.plot_network_clusters(testg, louvain, pos,figsize=(20,20),node_size=500,cmap='gist_rainbow', plot_labels=False)
#     nx.draw_networkx_edge_labels(testg,pos, edge_labels=labels, font_size=6)
#     nx.draw_networkx_labels(testg, pos, font_size=8)
#     # viz.plot_community_graph(testg,louvain)
#     plt.savefig("Louvain_10com.png")
#     plt.show()
#     plt.close()

 
    # graphs = (testg.subgraph(c) for c in nx.connected_components(testg))
    # graphs = list(graphs)
    # print(graphs)
    # community = weight_fluid(my_graph2, 10, seed=1)
    
    # fluid = nx.algorithms.community.asyn_fluidc(testg, 16, seed=10)
    # defaultFLuid = algorithms.async_fluid(my_graph2, 13)
    # louvain = algorithms.louvain(my_graph2, weight='weight')
    # com1 = []
    # com2 = []
    # coms1 = [list(x) for x in community]
    # fluid2 = [list(x) for x in fluid]
    # coms2 = cdlib.NodeClustering(coms1, my_graph2, "FluidWeight")
    # fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")

    # Resoutions for 07/15 for Louvain to get 5,,7,10,13,15,17,20 communities
    # resolut = {}
    # resolut["5"] = 4
    # resolut["7"] = 2.5
    # resolut["10"] = 1.41
    # resolut["13"] = 1
    # resolut["15"] = 0.9
    # resolut["17"] = 0.72
    # resolut["20"] = 0.6
    # resolutions = [4,2.5,1.41,1,0.9,0.72,0.6]
    # louvain = algorithms.louvain(testg, weight='weight', resolution=0.4)

    # count = 0
    # for i in fluid3.communities:
    #     count = count +1
    # print(count)

    # count = 0
    # for i in louvain.communities:
    #     count = count +1
    # print(count)

    # with open('algo1fluidcontrol5comm.txt', 'w') as f:

    #     scores = []
    #     for s in range(30):
    #         print("seed: "+ str(s))
    #         print("seed: "+ str(s),file=f)
            # wcom = weight_fluid(testg, 5, seed=s)
            # wcoms = [list(x) for x in wcom]
            # wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
            # fluid = nx.algorithms.community.asyn_fluidc(testg, 5, seed=s)
            # fluid2 = [list(x) for x in fluid]
            # fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")
    #         print("weightedfluid")
    #         print(wcoms2.communities)
    #         print("Benchmark Fluid")
    #         print(fluid3.communities)
    #         print("weightedfluid", file=f)
    #         print(wcoms2.communities, file=f)
    #         print("Benchmark Fluid", file=f)
    #         print(fluid3.communities,file=f)
    #         scores.append(evaluation.adjusted_rand_index(wcoms2, fluid3).score)
    #         print(evaluation.adjusted_rand_index(wcoms2, fluid3), file=f)
    #     print("Adjusted rand indexes")
    #     print("Adjusted rand indexes", file=f)
    #     print(scores)
    #     print(scores, file=f)
    #     print("Mean")
    #     print("Mean", file=f)
    #     print(numpy.mean(scores))
    #     print(numpy.mean(scores),file=f)
    #     print("Standard deviation")
    #     print("Standard deviation", file=f)
    #     print(numpy.std(scores))
    #     print(numpy.std(scores), file=f)

    # with open('algo1louvainnorand5comm.txt', 'w') as f:

    #     scores = []
    #     for s in range(30):
    #         print("seed: "+ str(s))
    #         print("seed: "+ str(s),file=f)
    #         wcom = weight_fluid(testg, 5, seed=s)
    #         wcoms = [list(x) for x in wcom]
    #         wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
    #         louvain = algorithms.louvain(testg, weight='weight', resolution=15)
    #         print("weightedfluid")
    #         print(wcoms2.communities)
    #         print("Benchmark Fluid")
    #         print(louvain.communities)
    #         print("weightedfluid", file=f)
    #         print(wcoms2.communities, file=f)
    #         print("Benchmark Fluid", file=f)
    #         print(louvain.communities,file=f)
    #         print(evaluation.adjusted_rand_index(wcoms2, louvain))
    #         print(evaluation.adjusted_rand_index(wcoms2, louvain),file=f)
    #         scores.append(evaluation.adjusted_rand_index(wcoms2, louvain).score)
    #     print("Adjusted rand indexes")
    #     print("Adjusted rand indexes", file=f)
    #     print(scores)
    #     print(scores, file=f)
    #     print("Mean")
    #     print("Mean", file=f)
    #     print(numpy.mean(scores))
    #     print(numpy.mean(scores),file=f)
    #     print("Standard deviation")
    #     print("Standard deviation", file=f)
    #     print(numpy.std(scores))
    #     print(numpy.std(scores), file=f)

    # with open('algo1louvainrand5comm.txt', 'w') as f:
    #     scores = []
    #     for s in range(30):
    #         print("seed: "+ str(s))
    #         print("seed: "+ str(s),file=f)
    #         wcom = weight_fluid(testg, 5, seed=s)
    #         wcoms = [list(x) for x in wcom]
    #         wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
    #         #Adjust resolution to get community size [4,2.5,1.41,1,0.9,0.72,0.6] -> [5,7,10,13,15,17,20] for 07/15 [14,3.5,1.5,1,0.7,0.6,0.4] for 05/06
    #         louvain = algorithms.louvain(testg, weight='weight',randomize=1, resolution=14)
    #         print("weightedfluid")
    #         print(wcoms2.communities)
    #         print("Benchmark Fluid")
    #         print(louvain.communities)
    #         print("weightedfluid", file=f)
    #         print(wcoms2.communities, file=f)
    #         print("Benchmark Fluid", file=f)
    #         print(louvain.communities,file=f)
    #         print(evaluation.adjusted_rand_index(wcoms2, louvain))
    #         print(evaluation.adjusted_rand_index(wcoms2, louvain),file=f)
    #         scores.append(evaluation.adjusted_rand_index(wcoms2, louvain).score)
    #     print("Adjusted rand indexes")
    #     print("Adjusted rand indexes", file=f)
    #     print(scores)
    #     print(scores, file=f)
    #     print("Mean")
    #     print("Mean", file=f)
    #     print(numpy.mean(scores))
    #     print(numpy.mean(scores),file=f)
    #     print("Standard deviation")
    #     print("Standard deviation", file=f)
    #     print(numpy.std(scores))
    #     print(numpy.std(scores), file=f)

        # for s in range(1, 90, 3):
        #     print("seed: "+ str(s))
        #     # wcom = weight_fluid(my_graph2, 10, seed=s)
        #     # wcoms = [list(x) for x in wcom]
        #     # wcoms2 = cdlib.NodeClustering(wcoms, my_graph2, "FluidWeight")
        #     louvain = algorithms.louvain(my_graph2, weight='weight', randomize=s, resolution=1.41)
        #     fluid = nx.algorithms.community.asyn_fluidc(my_graph2, 10, seed=s)
        #     fluid2 = [list(x) for x in fluid]
        #     fluid3 = cdlib.NodeClustering(fluid2, my_graph2, "FluidWeight")
        #     print(evaluation.adjusted_rand_index(fluid3, louvain))
    
