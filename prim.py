#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Prim's algorithm for creation of a minimun spanning tree

@author: eduardo
"""

import matplotlib.pyplot as plt
import numpy as np
from kruskal import create_points, distance, distances, sort_edges, \
    get_nodes, get_indexes


def get_dists(points, nodes=None):
    """ Get the distances among a set of points, creates edges and determine
    its weight according the distance between its nodes
    :param points, a list of Point objects """
    d = {}
    for i in range(len(points)):
        for j in range(len(points)):
            # Avoid both, parallel edges (creation of links that already exist)
            # and loops (link a point with itself)
            dist = distance(points[i], points[j])
            if i < j:
                if nodes is None:
                    d['{0}-{1}'.format(i, j)] = dist
                else:
                    d['{0}-{1}'.format(nodes[i], nodes[j])] = dist
    return d


def base_distances(points, nodes=None):
    """ Get the distances among a node and its possible connections according
    the predefined base graph.
    :param points, a list of Point objects
    :param nodes, the nodes of base graph"""
    d = {}
    for i in range(1, len(points)):
        dist = distance(points[0], points[i])
        if nodes is None:
            d['{0}-{1}'.format(0, i)] = dist
        else:
            d['{0}-{1}'.format(nodes[0], nodes[i])] = dist
    return d


def not_prim(points, base):
    """ Prim's algorithm for Minimum Spanning Tree """

    nodes = len(points)
    for i in range(nodes):
        print('*** NODE: {0} ***'.format(i))
        this_points = []
        this_nodes = []
        this_points.append(points[i])
        this_nodes.append(i)
        for j in base[i]:
            this_points.append(points[j])
            this_nodes.append(j)
        print('Nodes: ', this_nodes)
        # Get the distances of edges among the nodes and sort them
        d = base_distances(this_points, this_nodes)
        edges, dist = sort_edges(d)
        for e, d in zip(edges, dist):
            print("{0}:\t{1:0.2f}".format(e, d))
    # Vertices not yet included in the tree
    p = range(nodes)
    init = int(np.random.choice(p, 1))
    print('Selected point: {0}'.format(init))
    # Initialize the tree with a single vertex chosen randomly
    tree = [init]
    return tree



def prim(points):
    """ Prim's algorithm for Minimum Spanning Tree """

    nodes = len(points)
    d = get_dists(points)
    edges, dist = sort_edges(d)
    for e, d in zip(edges, dist):
        print("{0}:\t{1:0.2f}".format(e, d))
    tree = []
    tree_nodes = []
    # Vertices not yet included in the tree
    p = range(nodes)
    init = int(np.random.choice(p, 1))
    print('Selected point: {0}'.format(init))
    edge = get_nodes(edges[init])
    print(edge)
    plt.plot([points[edge[0]].x, points[edge[1]].x],
             [points[edge[0]].y, points[edge[1]].y])
    tree.append(edge)
    tree_nodes.append(edge[0])
    tree_nodes.append(edge[1])
    p.remove(edge[0])
    p.remove(edge[1])

    while len(p) > 0:
        for e in edges:
            edge = get_nodes(e)
            indexes = get_indexes(edge, tree)
            if indexes[0] is None and indexes[1] is None:
                print('Edge: [{0},{1}] doesnt touch tree'.format(edge[0],
                      edge[1]))
            elif indexes[0] is not None and indexes[1] is None:
                print('Edge: [{0},{1}] touchs tree. Adding {1}'.format(edge[0],
                      edge[1]))
                tree.append(edge)
                tree_nodes.append(edge[1])
                p.remove(edge[1])
                break
            elif indexes[0] is None and indexes[1] is not None:
                print('Edge: [{0},{1}] touchs tree. Adding {0}'.format(edge[0],
                      edge[1]))
                tree.append(edge)
                tree_nodes.append(edge[0])
                p.remove(edge[0])
                break
            elif indexes[0] != indexes[1]:
                print('Edge: [{0},{1}] already in tree'.format(edge[0],
                      edge[1]))
            else:
                print('Edge: [{0},{1}] will be ignored'.format(edge[0],
                      edge[1]))
        print(tree)
        print(tree_nodes)
    return tree


if __name__ == '__main__':

    # Coordinates from Angeles (2002) network
    plt.title("Nodes from Angeles (2002)")
    # These coordinates include an additonal node: '12'
    # cx = [626.33, 586.33, 261.33, 514.74, 176.67, 419.91, 0.0, 141.18, 360.86,
    #       274.27, 157.32, 425.24, 100.79]
    # cy = [743.17, 743.17, 743.17, 567.16, 542.28, 391.07, 462.77, 305.57,
    #       216.80, 93.29, 0.0, 11.67, 401.71]
    cx = [626.33, 586.33, 261.33, 514.74, 176.67, 419.91, 0.0, 141.18, 360.86,
          274.27, 157.32, 425.24]
    cy = [743.17, 743.17, 743.17, 567.16, 542.28, 391.07, 462.77, 305.57,
          216.80, 93.29, 0.0, 11.67]
    base_graph = [[1, 3],
                  [2, 3],
                  [3, 4],
                  [2, 5],
                  [2, 5, 6, 7],
                  [3, 4, 8],
                  [4, 7],
                  [4, 6, 8, 9],
                  [5, 7, 9, 11],
                  [7, 8, 10, 11],
                  [7, 9],
                  [8, 9]]
    # Coordinates from Geem et al. (2000) network
    # plt.title("Nodes from Geem et al. (2000)")
    # cx = [0, 100, 200, 0, 100, 200, 0, 100, 200]
    # cy = [0, 0, 0, 100, 100, 100, 200, 200, 200]

    # Create the points objects
    points = create_points(cx, cy)

    mst = prim(points)

    # Plot the nodes of the network
    plt.scatter(cx, cy)
    labels = ['{0}'.format(i) for i in range(len(cx))]
    for label, x, y in zip(labels, cx, cy):
        plt.annotate(label, xy=(x, y), xytext=(-5, 5),
                     textcoords='offset points')
#    for j in range(len(base_graph)):
#        for k in base_graph[j]:
#            plt.plot([points[j].x, points[k].x],
#                     [points[j].y, points[k].y])
    # Print the MST
    for n in mst:
        plt.plot([points[n[0]].x, points[n[1]].x],
                 [points[n[0]].y, points[n[1]].y])
    plt.show()
