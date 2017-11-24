#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Kruskal algorithm

@author: eduardo
"""

from math import sqrt
import matplotlib.pyplot as plt


class Point:
    """ Define a point object """
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


def create_points(x, y):
    """ Create a set of Points objects from two lists of coordinates (x,y)
    :param x, a list with the X coordinates of the points
    :param y, a list with the Y coordinates of the points"""
    assert type(x) is list and type(y) is list, "Arguments should be lists"
    assert len(x) == len(y), "Lists should have same size"
    points = []
    for i in range(len(x)):
        points.append(Point(x[i], y[i]))
    return points


def distance(point1, point2):
    """ Get distance between two points
    :param point1, the first Point object
    :param point2, the second Point object"""
    dist = sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2 +
                (point2.z - point1.z)**2)
    return dist


def distances(points):
    """ Get the distances among a set of points, creates edges and determine
    its weight according the distance between its nodes
    :param points, a list of Point objects """
    d = {}
    for i in range(len(points)):
        for j in range(len(points)):
            # Avoid both, parallel edges (creation of links that already exist)
            # and loops (link a point with itself)
            if i < j:
                d['{0}-{1}'.format(i, j)] = distance(points[i], points[j])
    return d


def sort_edges(d):
    """ Sorts the links or edges according its weight
    :param d, a dictionary with edges as keys and distances as values"""
    edges = []
    dist = []
    # This block of code is tricky!
    for key, value in sorted(d.iteritems(), key=lambda (k, v): (v, k)):
        edges.append(key)
        dist.append(value)
    return edges, dist


def get_nodes(s):
    chars = s.split('-')
    nodes = [int(x) for x in chars]
    return nodes


def get_indexes(nodes, mlist):
    """ Get indexes
    Those indexes indicate in which set or subtree the nodes are
    located.
    """
    assert type(nodes) is list, "Argument should be a list"
    assert type(mlist) is list, "Argument should be a list"
    assert len(nodes) == 2, "List should have two elements"

    indexes = [None, None]
    inode = nodes[0]  # Initial node
    fnode = nodes[1]  # Final node
    for i in range(len(mlist)):
        if inode in mlist[i]:
            indexes[0] = i
        if fnode in mlist[i]:
            indexes[1] = i
    return indexes


def kruskal(points):
    """ Kruskal's algorithm

    Genetates the Minimum Spanning Tree (MST) with the coordinates from a set
    of nodes.
    """
    nodes = len(points)
    # Get all the distances among the points, these will be the weights for
    # Kruskal's algorthim
    # This will create all the possible edges among the nodes
    w = distances(points)

    # Sort the links (a.k.a. edges) by weight
    edges, dist = sort_edges(w)
    for e, d in zip(edges, dist):
        print("{0}:\t{1:0.2f}".format(e, d))
    # Create a subtree for each node
    subtrees = []
    for i in range(nodes):
        subtrees.append([])

    # Generate the minimum spanning tree (MST)
    mst = []
    i = 0       # Counter to add distance
    length = 0  # Total length of the network
    for edge in edges:
        # Get both nodes of the edge, the initial and final nodes
        # WARNING! this list should always have a length of two
        edge_nodes = get_nodes(edge)
        inode = edge_nodes[0]
        fnode = edge_nodes[1]

        # Find in which subtrees are the nodes of the edge
        tree_indexes = get_indexes(edge_nodes, subtrees)

        # Add the current edge to the MST if the nodes are in different sets
        # (or subtrees), this means that both indexes are equal to 'None' or
        # they have different values.
        if tree_indexes[0] is None and tree_indexes[1] is None:
            index = min(edge_nodes)
            # Populate the subtree of the min index
            print("Edge ({0},{1}): adding to subtree {2} ({3:0.2f})".format(
                    inode, fnode, index, dist[i]))
            subtrees[index].append(inode)
            subtrees[index].append(fnode)
            mst.append(edge_nodes)
            length += dist[i]
        elif tree_indexes[0] is not None and tree_indexes[1] is None:
            index = tree_indexes[0]
            print("Edge ({0}*,{1}): adding to subtree {2} ({3:0.2f})".format(
                    inode, fnode, index, dist[i]))
            subtrees[index].append(fnode)
            mst.append(edge_nodes)
            length += dist[i]
        elif tree_indexes[0] is None and tree_indexes[1] is not None:
            index = tree_indexes[1]
            print("Edge ({0},{1}*): adding to subtree {2} ({3:0.2f})".format(
                    inode, fnode, index, dist[i]))
            subtrees[index].append(inode)
            mst.append(edge_nodes)
            length += dist[i]
        elif tree_indexes[0] != tree_indexes[1]:
            # Be sure the indexes have different values
            index = min(tree_indexes)
            empty = max(tree_indexes)
            print("Edge ({0},{1}): merging trees {2} & {3} ({4:0.2f})".format(
                    inode, fnode, tree_indexes[0], tree_indexes[1], dist[i]))
            subtrees[index] = subtrees[index] + subtrees[empty]
            subtrees[empty] = []
            mst.append(edge_nodes)
            length += dist[i]
        else:
            print("Edge ({0},{1}): ignoring.".format(inode, fnode))
        # Check termination criteria
        if len(mst) == (nodes - 1):
            break
        i += 1
    return mst, length

if __name__ == '__main__':

    # Coordinates from Angeles (2002) network
    # These coordinates include an additonal node: '12'
#    cx = [626.33, 586.33, 261.33, 514.74, 176.67, 419.91, 0.0, 141.18, 360.86,
#          274.27, 157.32, 425.24, 100.79]
#    cy = [743.17, 743.17, 743.17, 567.16, 542.28, 391.07, 462.77, 305.57,
#          216.80, 93.29, 0.0, 11.67, 401.71]
#    cx = [626.33, 586.33, 261.33, 514.74, 176.67, 419.91, 0.0, 141.18, 360.86,
#          274.27, 157.32, 425.24]
#    cy = [743.17, 743.17, 743.17, 567.16, 542.28, 391.07, 462.77, 305.57,
#          216.80, 93.29, 0.0, 11.67]
     # Coordinates from Geem et al. (2000) network
    cx = [0, 100, 200, 0, 100, 200, 0, 100, 200]
    cy = [0, 0, 0, 100, 100, 100, 200, 200, 200]

    # nodes = len(cx)  # Or len(cy)
    # Create the points objects
    points = create_points(cx, cy)

    # Get the Minimum Spanning Tree with Kruskal's algorithm
    mst, length = kruskal(points)
    print("Total length of the network: {0:0.2f}".format(length))

    # Plot the nodes of the network
    fig1 =  plt.figure(1)
    plt.scatter(cx, cy)
    labels = ['{0}'.format(i) for i in range(len(cx))]
    for label, x, y in zip(labels, cx, cy):
        plt.annotate(label, xy=(x, y), xytext=(-5, 5),
                     textcoords='offset points')
    # Print all combinations
#    comb = 0
#    for i in range(nodes):
#        for j in range(nodes):
#            # Avoid both previous links and to link a point with itself
#            if i < j:
#                plt.plot([points[i].x, points[j].x],
#                         [points[i].y, points[j].y])
#                comb += 1
#    print(comb)

    # Print the MST
    for n in mst:
        plt.plot([points[n[0]].x, points[n[1]].x],
                 [points[n[0]].y, points[n[1]].y])

    #plt.title("Nodes from Angeles (2002), length: {0:0.2f}".format(length))
    plt.title("MST for irrigation network")
    plt.savefig("kruskal.png", bbox_inches='tight', dpi=150)
    plt.show()
