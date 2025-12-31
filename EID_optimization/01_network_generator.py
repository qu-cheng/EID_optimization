import numpy as np
import numpy.random as rdm
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import sys


def heterogeneous_dist(ID_list, mean_degree, heterogeneity):
    # intiate with all nodes having mean degree
    degree = {}
    for ID in ID_list:
        degree[ID] = mean_degree

    # iterate until heterogeneity is achieved
    while np.std([degree[ID] for ID in ID_list]) < heterogeneity:
        # for i in range(switches):
        # choose a node at random
        source_list = [ID for ID in ID_list if degree[ID] > 1]
        random_node = source_list[int(rdm.random() * len(source_list))]

        # select a node with probability proportional to its degree
        # 1. get a random number that corresponds to an edge
        r = rdm.random() * len(ID_list) * mean_degree

        # 2. find which node the random number corresponds to
        k = 0
        # start at the start of the list and slide up until random number is reached
        target_node = ID_list[k]
        slide = degree[target_node]
        while slide < r:
            k = k + 1
            target_node = ID_list[k]
            slide = slide + degree[target_node]

        # remove the stub from the random node and attach to the target
        degree[random_node] -= 1
        degree[target_node] += 1

    return degree


def connect_stubs(stubs, edge_list):
    # randomize the list
    stubs = [(s[0], s[1]) for s in rdm.permutation(stubs)]

    # new_edges=[]
    if len(stubs) < 2:
        no_more_edges = True
    else:
        no_more_edges = False

    while no_more_edges == False:
        # take the first stub from the list
        source = stubs.pop(0)

        # print('Source:',source)
        found = False
        stubs_to_check = len(stubs)

        # loop until a stub has been found
        while found == False and no_more_edges == False:
            # take the first form the list
            target = stubs.pop(0)

            # order the edge so that the reverse one dosen't get put in
            new_edge = sorted([source, target])
            if new_edge in edge_list or source == target:
                # if not then throw it back in the list
                stubs.append(target)

            # otherwise create the edge
            else:
                edge_list.append(new_edge)
                found = True

            # one less edge to check so deduct from n
            stubs_to_check = stubs_to_check - 1
            if stubs_to_check < 2:
                no_more_edges = True

    return edge_list


def modular_config_model(module_size, number_of_modules, p, heterogeneity, mean_degree):
    ID_list = []
    # create list of IDs according to the (module,number) naming convention
    for m in range(number_of_modules):
        for n in range(module_size):
            node_ID = (m, n)
            ID_list.append(node_ID)

    degree = heterogeneous_dist(ID_list, mean_degree, heterogeneity)

    # for the degee distribution
    edge_list = []
    # these are the stubs that connect together across modules
    inter_stubs = []

    for m in range(number_of_modules):
        # first create a list of stubs within the module
        intra_stubs = []
        for n in range(module_size):

            for i in range(degree[(m, n)]):
                r = rdm.random()
                if r < p:
                    intra_stubs.append((m, n))
                else:
                    inter_stubs.append((m, n))

        # for the intra stubs
        new_edges = connect_stubs(intra_stubs, edge_list)

        edge_list = edge_list + new_edges

    # now do the same for the inter stubs
    new_edges = connect_stubs(inter_stubs, edge_list)

    #    new_edges=[[tuple(stubs[i]),tuple(stubs[total_degree-1-i])] for i in range(int(total_degree/2))]

    edge_list = edge_list + new_edges

    return ID_list, edge_list

def network_generator(module_size, number_of_modules, p, heterogeneity, mean_degree):
    ID = []
    edge = []
    ID2, edge2 = modular_config_model(module_size, number_of_modules, p, heterogeneity, mean_degree)
    for i in ID2:
        ID.append(str(i))
    for i in edge2:
        x = []
        for j in i:
            j = tuple(map(int, j))
            x.append(str(j))
        edge.append(x)
    return ID,edge
