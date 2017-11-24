# -*- coding: utf-8 -*-
"""
objfunctions.py

Objective functions for pipe networks optimization

Epanet (Python wrappers) is used to run the hydraulic simulations required to
evaluate the fitness of the individual networks.

@author: ecoslacker
"""
import csv
import numpy as np


def generate_limits(dim, lb, ub):
    """ Generate limits

    Generate two vectors of the specified dimension, one with the lower
    boundary values and the other with upper boundary values.

    :param dim, the dimension of the vectors (list)
    :param lb, lower boundary value
    :param ub, upper boundary value
    :return lblist, ublist, two lists with the lower and upper boundaries
    """
    assert type(dim) is int, "Dimension should be integer"
    assert dim > 0, "Dimension should be greater than zero"
    assert type(lb) is int or type(lb) is float, "lb should be numeric"
    assert type(ub) is int or type(ub) is float, "lb should be numeric"
    assert lb < ub, "Lower boundary is greater or equal than upper value"

    lblist = [lb] * dim
    ublist = [ub] * dim
    return lblist, ublist


def is_feasible(x, lb, ub):
    """ Is feasible

    Check is every value of the list "x" is between lower boundary and
    upper boundary.

    :param x, a list of values
    :param lb, a list or value for lower boundaries
    :param ub, a list or value for upper boundaries
    :return True, if lb < x < ub
    """
    assert type(x) is list, "First parameter should be a list of values"
    assert len(x) > 0, "First parameter is an empty list"
    assert (type(lb) is int and type(ub) is int) or \
        (type(lb) is float and type(ub) is float) or \
        (type(lb) is list and type(ub) is list), \
        "Type of boundaries don't match (Both should be int, float or list)"

    # Convert to numpy arrays
    xa = np.array(x)
    lba = np.array(lb)
    uba = np.array(ub)
    check = np.all(xa >= lba) and np.all(xa <= uba)
    return check


def unit_cost(diameter):
    """ Unit cost

    Returns the cost per unit of length for pipe of the specified diameter.

    Cost data obtained from:
      Xiaojun Zhou and David Y. Gao and Angus R. Simpson, 2016
      Optimal design of water distribution networks by a discrete state
      transition algorithm, Engineering Optimization Vol. 48, Num. 4, 603-628

    :param int diameter, index of the diameter in the list of available ones
    """
    assert type(diameter) is int, 'Required integer type'
    pipe_prices = {25: 2,
                   51: 5,
                   76: 8,
                   102: 11,
                   152: 16,
                   203: 23,
                   254: 32,
                   305: 50,
                   356: 60,
                   406: 90,
                   457: 130,
                   508: 170,
                   559: 300,
                   610: 550}
    return pipe_prices[diameter]


def unit_cost_meter(diameter):
    """ Unit cost

    Returns the cost per unit of length for pipe of the specified diameter.

    Cost data obtained from:
        Afshar, M. H., Akbari, M., & Mariño, M. A. (2005). Simultaneous layout
        and size optimization of water distribution networks: engineering
        approach. Journal of Infrastructure Systems, 11(4), 221–230.
        https://doi.org/10.1061/(ASCE)1076-0342(2005)11:4(221)

    :param int diameter, index of the diameter in the list of available ones
    """
    assert type(diameter) is int, 'Required integer type'
    pipe_prices = {125: 58,
                   150: 62,
                   200: 71.2,
                   250: 88.9,
                   300: 112.3,
                   350: 138.7,
                   400: 169,
                   450: 207,
                   500: 248,
                   550: 297,
                   600: 347,
                   650: 405,
                   700: 470}
    return pipe_prices[diameter]


def ind2arc(indexes, base):
    """ Indexes to arcs

    Retrieve a list of arcs from a base graph, according to the given indexes

    :param indexes, a list of integer indexes
    :param base, the base graph
    """
    arcs = []
    for i in indexes:
        arcs.append(base[i])
    return arcs


def distance(x1, y1, x2, y2):
    """ Distance

    Euclidean distance between two points.
    """
    d = 0
    return d


def read_instance(filename, base_graph):
    """ Read instance

    Reads the problem instance from a text delimited file, the file must
    be formatted as the *.tsp type as described in TSPLIB

    :param filename, a string containing the name of the file with data
    :param base_graph, True to open a text file as base graph for the nodes
    :return x, the x-axis coordinates of the instance
    :return y, the y-axis coordinates of the instance
    :return name, the name of the instance
    """
    labels = []
    x = []
    y = []
    z = []
    base = []
    name = ''
    read = False
    try:
        if base_graph:
            # Create the base graph file and read data from it
            base_file = filename[:-4] + "_base.csv"
            print("Using base graph data from: {0}".format(base_file))
            with open(base_file, 'r') as f:
                base = f.read().split('\n')
            if '' in base:
                base.remove('')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if row[0] == 'NAME':
                    name = row[2]
                if row[0] == 'COMMENT':
                    name = name + ' - ' + ' '.join(row[2:])
                if row[0] == 'NODE_COORD_SECTION':
                    read = True
                elif row[0] == 'EOF':
                    read = False
                elif read is True:
                    labels.append(row[0])
                    x.append(float(row[1]))
                    y.append(float(row[2]))
                    z.append(0.0)
    except IOError as e:
            z = e
            print(z)
    return x, y, z, name, labels, base
