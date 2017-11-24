#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Creates an open network using a simple Tree Growing Algorithm.
This kind of networks are used for irrigation.

@author: eduardo
"""

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv


def get_nodes(arc):
    """ Get nodes

    Get the initial and final nodes from an arc

    :param arc, a string arc with the nodes delimited by an "-"
    :return, an integer list with the arc nodes
    """
    assert type(arc) is str, 'Argument should be string'
    chars = arc.split('-')
    nodes = [int(x) for x in chars]
    return nodes


def adjacent(node, arcs):
    """ Adjacent

    Generates a list of adjacent arcs to the given node

    :param node, an integer node
    :param arcs, a list of integer nodes forming a network
    """
    adj = []  # Adjacent arcs
    for i in range(len(arcs)):
        nodes = get_nodes(arcs[i])
        if node in nodes:
            adj.append(i)
    return adj


def prob(items):
    """ Probabilities

    Creates a list of probabilities for a list of items. Intended to test
    the tree growing algorithm

    :param items, a list with the items
    """
    p = []
    for i in items:
        p.append(1.0 / len(items))
    return p


def tree_growing(x, y, base):
    """ Tree Growing Algorithm

    Tree Growing Algorithm from Walters & Smith (1995)

    :param x, list with X coordinates
    :param y, list with Y coordindates
    :param base, base graph of connections among nodes
    """
    total_nodes = len(x)
    print("Total nodes: {}".format(total_nodes))
    C = []   # set of nodes contained within the growing tree
    A = []   # set of arcs within the growing tree
    AA = []  # set of arcs adjacent to the growing tree

    # 1) Identify the root node Nr
    Nr = np.random.randint(total_nodes)
    print("Initial root node: {}".format(Nr))

    # 2) Initialise C = [Nr]
    C.append(Nr)

    # 3) Initialise A = [ ]
    # 4) Initialise AA = [arcs in base graph connected to root'node]
    AA = adjacent(Nr, base)
#    print('Arcs adjacent to node {0} are AA={1}'.format(Nr, AA))

    iters = 0
    while len(A) != (total_nodes - 1):
        # print('**** Iteration {0} ****'.format(iters))
        # print('  AA = {0}'.format(AA))
        # 5) Choose arc, a, at random from AA
        # probabilities = prob(AA)
        # print('  Probabilities={0}'.format(probabilities))
        # a = int(np.random.choice(AA, 1, p=probabilities))
        a = int(np.random.choice(AA, 1))
        # print('  Selected arc a={0} is ({1})'.format(a, base[a]))

        # 6) A = A + [a]
        A.append(a)
        # print('  A  = {0}'.format(A))

        # 7) Identify newly connected node, N
        # 8) C = C + [N]
        nodes = get_nodes(base[a])
        N = -1  # Just initialize with something dummy
        for node in nodes:
            if node not in C:
                N = node
                C.append(N)
        # print('  C  = {0}'.format(C))
        # 9) Identify arcs, ac(i), connected to N in base graph,
        #    (excluding arc a)
        aci = adjacent(N, base)
        if a in aci:
            # print('  Removing a from ac(i)')
            aci.remove(a)
        # print('  Adjacent to N={0} are ac(i)={1}'.format(N, aci))

        # 10) Update AA, by removing arc a and any newly infeasible arcs,
        #     and adding any of arcs ac(i) that are feasible candidates.
        #     AA now contains all feasible choices for the next arc of the tree

        # 10a) AA = AA - a (remove newly connected arc from list)
        AA.remove(a)

        # 10b) For each ac(i),
        #          Is ac(i) in AA already?
        #              Yes: AA = AA - [ac(i)] (remove ac(i) from list,
        #                                      as tree is now such that adding
        #                                      ac(i) would cause a loop)
        #              No: Are both end nodes of ac(i) in C?
        #                  Yes: AA = AA (leave list unaltered, as adding ac(i)
        #                               would cause a loop in the tree)
        #                  No: Is ac(i) in correct direction ?
        #                      Yes: AA = AA + [ac(i)] (add ac(i) to list)
        #                      No: AA = AA
        for ac in aci:
            if ac in AA:
                AA.remove(ac)
            else:
                nds = get_nodes(base[ac])
#                print('  Nodes of arc {0} are {1}'.format(ac, nds))
                if nds[0] in C and nds[1] in C:
                    # Do nothing
                    # print('  Keeping AA={0}'.format(AA))
                    dummy = 1
                else:
                    # Assuming correct direction
                    AA.append(ac)
#                    print('  Added {0} to AA={1}'.format(ac, AA))
        iters += 1
    print(A)
    return A


def open_coord(file_name):
    x = []
    y = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[1])
            y.append(row[2])
    return x, y


def open_base(file_name):
    x = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[0])
    return x

if __name__ == "__main__":

    # Coordinates for easiest example
    x = []
    y = []
    base = []
#    x = [0, 100, 0, 100]
#    y = [0, 0, 100, 100]
#    base = ['0-1', '0-2', '2-3', '1-3']

    # Coordinates of the second easiest problem
#    x = [0, 100, 200, 0, 100, 200]
#    y = [0, 0, 0, 100, 100, 100]
#    base = ['0-1', '1-2', '0-3', '1-4', '2-5', '3-4', '4-5']

#    # Coordinates of Angeles (2002) network
#    x = [626.33, 586.33, 261.33, 514.74, 176.67, 419.91, 0.0, 141.18, 360.86,
#         274.27, 157.32, 425.24]
#    y = [743.17, 743.17, 743.17, 567.16, 542.28, 391.07, 462.77, 305.57,
#         216.80, 93.29, 0.0, 11.67]
#    base = ['0-1', '1-2', '1-3', '2-4', '2-3', '3-5', '5-8', '4-6', '4-7',
#            '6-7', '7-8', '8-9', '9-10', '8-11', '9-11', '4-5', '2-5']

    # Coordinates of 8x8 grid network
    x = [0, 10, 0, 20, 10, 0, 30, 20, 10, 0, 40, 30, 20, 10, 0, 50, 40, 30,
         20, 10, 0, 60, 50, 40, 30, 20, 10, 0, 70, 60, 50, 40, 30, 20, 10, 0,
         70, 60, 50, 40, 30, 20, 10, 70, 60, 50, 40, 30, 20, 70, 60, 50, 40,
         30, 70, 60, 50, 40, 70, 60, 50, 70, 60, 70]
    y = [0, 0, 10, 0, 10, 20, 0, 10, 20, 30, 0, 10, 20, 30, 40, 0, 10, 20, 30,
         40, 50, 0, 10, 20, 30, 40, 50, 60, 0, 10, 20, 30, 40, 50, 60, 70, 10,
         20, 30, 40, 50, 60, 70, 20, 30, 40, 50, 60, 70, 30, 40, 50, 60, 70,
         40, 50, 60, 70, 50, 60, 70, 60, 70, 70]
    base = ['0-1', '1-3', '3-6', '6-10', '10-15', '15-21', '21-28',
            '0-2', '1-4', '3-7', '6-11', '10-16', '15-22', '21-29', '28-36',
            '2-4', '4-7', '7-11', '11-16', '16-22', '22-29', '29-36',
            '2-5', '4-8', '7-12', '11-17', '16-23', '22-30', '29-37', '36-43',
            '5-8', '8-12', '12-17', '17-23', '23-30', '30-37', '37-43',
            '5-9', '8-13', '12-18', '17-24', '23-31', '30-38', '37-44',
            '43-49', '9-13', '13-18', '18-24', '24-31', '31-38', '38-44',
            '44-49', '9-14', '13-19', '18-25', '24-32', '31-39', '38-45',
            '44-50', '49-54', '14-19', '19-25', '25-32', '32-39', '39-45',
            '45-50', '50-54', '14-20', '19-26', '25-33', '32-40', '39-46',
            '45-51', '50-55', '54-58', '20-26', '26-33', '33-40', '40-46',
            '46-51', '51-55', '55-58', '20-27', '26-34', '33-41', '40-47',
            '46-52', '51-56', '55-59', '58-61', '27-34', '34-41', '41-47',
            '47-52', '52-56', '56-59', '59-61', '27-35', '34-42', '41-48',
            '47-53', '52-57', '56-60', '59-62', '61-63', '35-42', '42-48',
            '48-53', '53-57', '57-60', '60-62', '62-63']

    # Two source network from Walters and Smith (1995)
    # Open from text file
#    x, y = open_coord('data/two_source_network.csv')
#    base = open_base('data/two_source_network_base.csv')

    # Run the growing tree algorithm
    tree = tree_growing(x, y, base)
    for arc in tree:
        print(base[arc])

    fig1 = plt.figure(1)
    plt.scatter(x, y)
    labels = ['{0}'.format(i) for i in range(len(x))]
    for label, lx, ly in zip(labels, x, y):
        plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                     textcoords='offset points')
#    for i in range(len(base)):
    for i in tree:
        n = get_nodes(base[i])
        plt.plot([ x[n[0]], x[n[1]] ],
                 [ y[n[0]], y[n[1]] ])

    # Uncomment for Walters and Smith (1995) network
#    for i in range(len(base)):
#        n = get_nodes(base[i])
#        plt.plot([ x[n[0]-1], x[n[1]-1] ],
#                 [ y[n[0]-1], y[n[1]-1] ])
    plt.show()

