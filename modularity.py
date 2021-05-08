import weightedfluid as wf1
import weightAlgo2 as wf2
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
import csv

if __name__ == '__main__': 

    name3 = "/home/james/4F90/sg_infectious_graphs/weightededgesX_2009_05_06.out"
    name = "/home/james/4F90/sg_infectious_graphs/weightededgesX_2009_07_15.out"
    fh2 = open(name, "rb")
    fh3 = open(name3, "rb")

    testg = nx.read_weighted_edgelist(fh3)
    # fh2.close()
    fh3.close()

    print(testg)

    fluid = nx.algorithms.community.asyn_fluidc(testg, 10, seed=5)
    fluid2 = [list(x) for x in fluid]
    fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")

    wcom = wf1.weight_fluid(testg, 9, seed=5)
    wcoms = [list(x) for x in wcom]
    wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
    print("Alg1")
    print(evaluation.newman_girvan_modularity(testg, wcoms2).score)

    # louvain = algorithms.louvain(testg, weight='weight', resolution=1.5)

    reso = [14,3.5,1.5,1,0.7,0.6,0.4]
    comNumb = [5,7,10,13,15,17,20]

    with open('modularity.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        print("Louvain")
        writer.writerow(["05/06"])
        writer.writerow(["Louvain"])
        writer.writerow(["Communities", "Modularity"])
        for r, c in zip(reso, comNumb):
            scores = []
            scores.append(c)
            for s in range(30):
                louvain = algorithms.louvain(testg, weight='weight', resolution=r)
                print("Communities: "+ str(c))
                rslt = evaluation.newman_girvan_modularity(testg, louvain).score
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("\n")
        writer.writerow("\n")
        print("Fluid")
        writer.writerow(["Fluid"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            scores = []
            scores.append(r)
            for s in range(30):
                fluid = nx.algorithms.community.asyn_fluidc(testg, r, seed=s)
                fluid2 = [list(x) for x in fluid]
                fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")
                rslt = evaluation.newman_girvan_modularity(testg, fluid3).score
                print("Communities: "+ str(r))
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("Algorithm 1")
        writer.writerow("\n")
        writer.writerow(["Algorithm 1"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            scores = []
            scores.append(r)
            for s in range(30):
                wcom = wf1.weight_fluid(testg, r, seed=s)
                wcoms = [list(x) for x in wcom]
                wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
                rslt = evaluation.newman_girvan_modularity(testg, wcoms2).score
                print("Communities: "+ str(r))
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("Algorithm 2")
        writer.writerow("\n")
        writer.writerow(["Algorithm 2"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            count = 0
            s = 0
            scores = []
            scores.append(r)
            while count <30:
                try:
                    wcom = wf2.asyn_fluidcWeight(testg, r, seed=s)
                    wcoms = [list(x) for x in wcom]
                    wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
                    rslt = evaluation.newman_girvan_modularity(testg, wcoms2).score
                    print("Communities: "+ str(r))
                    print(rslt)
                    scores.append(rslt)
                    count+=1
                    s+=1
                except:
                    s+=1
            writer.writerow(scores)
###############
        testg = nx.read_weighted_edgelist(fh2)
        fh2.close()
        print("Louvain")
        writer.writerow(["07/15"])
        writer.writerow(["Louvain"])
        writer.writerow(["Communities", "Modularity"])
        for r, c in zip(reso, comNumb):
            scores = []
            scores.append(c)
            for s in range(30):
                louvain = algorithms.louvain(testg, weight='weight', resolution=r)
                print("Communities: "+ str(c))
                rslt = evaluation.newman_girvan_modularity(testg, louvain).score
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("\n")
        writer.writerow("\n")
        print("Fluid")
        writer.writerow(["Fluid"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            scores = []
            scores.append(r)
            for s in range(30):
                fluid = nx.algorithms.community.asyn_fluidc(testg, r, seed=s)
                fluid2 = [list(x) for x in fluid]
                fluid3 = cdlib.NodeClustering(fluid2, testg, "FluidWeight")
                rslt = evaluation.newman_girvan_modularity(testg, fluid3).score
                print("Communities: "+ str(r))
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("Algorithm 1")
        writer.writerow("\n")
        writer.writerow(["Algorithm 1"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            scores = []
            scores.append(r)
            for s in range(30):
                wcom = wf1.weight_fluid(testg, r, seed=s)
                wcoms = [list(x) for x in wcom]
                wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
                rslt = evaluation.newman_girvan_modularity(testg, wcoms2).score
                print("Communities: "+ str(r))
                print(rslt)
                scores.append(rslt)
            writer.writerow(scores)
        print("Algorithm 2")
        writer.writerow("\n")
        writer.writerow(["Algorithm 2"])
        writer.writerow(["Communities", "Modularity"])
        for r in comNumb:
            count = 0
            s = 0
            scores = []
            scores.append(r)
            while count <30:
                try:
                    wcom = wf2.asyn_fluidcWeight(testg, r, seed=s)
                    wcoms = [list(x) for x in wcom]
                    wcoms2 = cdlib.NodeClustering(wcoms, testg, "FluidWeight")
                    rslt = evaluation.newman_girvan_modularity(testg, wcoms2).score
                    print("Communities: "+ str(r))
                    print(rslt)
                    scores.append(rslt)
                    count+=1
                    s+=1
                except:
                    s+=1
            writer.writerow(scores)

    # writer.close()
