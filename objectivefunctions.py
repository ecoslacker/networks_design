# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:17:08 2016

@author: ecoslacker
"""
import random
import numpy as np

from math import sqrt

from utilities import ind2arc
from tree_growing import get_nodes, adjacent

random.seed()

defaults = {25: 2,
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


def init_diameters2(x, dimension, *args):
    """ Initialize diameters

    Create a random list of diameters, one for each pipe of the network,
    diamerters should be in milimeters

    :param x, a list of diameters for each pipe of the network
    :param dimension, the number of pipes of tje network
    """
    diameters = [25, 51, 76, 102, 152, 203, 254, 305, 356, 406, 457, 508,
                 559, 610]
    for i in range(dimension):
        x.append(diameters[random.randrange(0, len(diameters))])
    return x


def init_diameters(x, dimension, **kwargs):
    prices = kwargs.get('price', defaults)  # Price per diameter
    diameters = prices.keys()
    diameters.sort()
    for i in range(dimension):
        x.append(diameters[random.randrange(0, len(diameters))])
    return x


def init_layout(x, dimension, **kwargs):
    """ Initialize function

    Creates a random list of numbers, that represent an open network layout

    :param x, a list that represent the connections amog the nodes
    :param dimension, the number of nodes of the network
    """
    for i in range(dimension):
        x.append(random.randrange(0, dimension))


def network_size_cost(networkobject, x, **kwargs):
    """" The objective function for dimensioning optimization

    Objective function from Afshar & Jabbari (2007)

    :param Network networkobject, an implementation of a 'Network' class
    :param list x, a set of pipe diameters
    :param dict price, pipe prices per length unit for each diameter
    :param float penalty, penalty value for infeasible networks
    :param float h_min, penalty value for infeasible networks
    :param float penalty, penalty value for infeasible networks
    :param float penalty, penalty value for infeasible networks
    :param float penalty, penalty value for infeasible networks
    :return fitness, minimum is better
    """
    unit_prices = kwargs.get('price', defaults)
    alpha = kwargs.get('penalty', 600000)  # Penalty parameter
    _hmin = kwargs.get('hmin', 30.0)
    _hmax = kwargs.get('hmax', 50.0)
    _vmin = kwargs.get('vmin', 0.5)
    _vmax = kwargs.get('vmax', 2.50)

    # Perform the EPANET simulation
    networkobject.change_diameters(x)
    p, v, pl = networkobject.simulate()

    # Convert the results to numpy arrays
    hmin = np.array([_hmin] * len(p), dtype=np.float)
    hmax = np.array([_hmax] * len(p), dtype=np.float)
    vmin = np.array([_vmin] * len(v), dtype=np.float)
    vmax = np.array([_vmax] * len(v), dtype=np.float)
    head = np.array(p, dtype=np.float)
    vel = np.array(v, dtype=np.float)

    # Compute the penalty cost
    penalty_cost = sum(1 - vel/vmin) + sum(vel/vmax - 1) + sum(1 - head/hmin)
    print('\n  Penalty cost: {0}'.format(penalty_cost))
    # \ + sum((head/hmax - 1)**2)

    # Objective function, cost from pipe length
    cost = 0
    for i in range(len(x)):
        # Real cost
        c = unit_prices[x[i]] * pl[i]
        cost += c
    print('  Cost: {0}'.format(cost))

    # Check for condition violations
    violations = sum(hmin > head) + sum(vmin > vel) + sum(vmax < vel)
    if violations == 0:
        alpha = 0
    print('  Violations: {0}'.format(violations))
    print('  Final cost: {0}'.format(cost + alpha * penalty_cost))

    return cost + alpha * penalty_cost


#def network_size2(networkobject, x, **kwargs):
#    """" The objective function for dimensioning optimization
#
#    :param networkobject, an implementation of a 'Network' class
#    :param x, a list of pipe diameters
#    :return fitness, minimum is better
#    """
#    unit_prices = kwargs.get('price', defaults)
#    _hmin = kwargs.get('hmin', 30.0)
#    _hmax = kwargs.get('hmax', 100.0)
#    _vmin = kwargs.get('vmin', 0.25)
#    _vmax = kwargs.get('vmax', 3.50)
#    _penalty = kwargs.get('penalty_step', 1)
#
#    # Perform the EPANET simulation
#    networkobject.change_diameters(x)
#    p, v, l = networkobject.simulate()
#
#    # print("Pressure: ", p)
#    # print("Velocity: ", v)
#
#    # Penalty for not supplying the required minimum pressure at each node
#    pH = 0
#
#    # TODO: Find a way to exclude certain nodes
#    # WARNING: This block will exclude the last node from HEAD penalty
#    # because for EPANET, last node is the SOURCE or TANK of the network and
#    # its pressure is always zero because is open to atmospheric pressure
#    # print('  Pressure:')
#    for i in range(len(p)-1):
#        # print('  {0} < {1}?'.format(p[i], _hmin))
#        if p[i] < _hmin:
#            # print('  yes')
#            pH += _penalty
#        if p[i] < 0:
#            pH += _penalty
#
#    # Penalty for violating the flow velocity limits of the pipes
#    pV = 0
#    # print('  Velocity:')
#    for i in range(len(v)):
#        # print('  {0} < {1} or {0} > {2}?'.format(v[i], _vmin, _vmax))
#        if v[i] < _vmin or v[i] > _vmax:
#            # print('  yes')
#            pV += _penalty
#
#    # This was added later
#    penalties = pV + pH
#    if penalties == 0:
#        # print('   No penalties')
#        penalties = 1
#
#    # Objective function
#    cost = 0
#    if pV == 0:
#        pV = 1
#    if pH == 0:
#        pH = 1
#    for i in range(len(x)):
#        # c = unit_prices[x[i]] * l[i]
#        c = unit_prices[x[i]] * l[i] * pV * pH
#        cost += c
#    # print('  Total cost: {0}, penalty: {1}'.format(cost, penalties))
#    # return cost * penalties
#    return cost


def network_size_aco(networkobject, x, **kwargs):
    """" The objective function for dimensioning optimization

    :param networkobject, an implementation of a 'Network' class
    :param x, a list of pipe diameters
    :return fitness, minimum is better
    """
    unit_prices = kwargs.get('price', defaults)
    _hmin = kwargs.get('hmin', 30.0)
    _hmax = kwargs.get('hmax', 100.0)
    _vmin = kwargs.get('vmin', 0.25)
    _vmax = kwargs.get('vmax', 4.00)
    _penalty = kwargs.get('penalty_step', 1)

    # Perform the EPANET simulation
    networkobject.change_diameters(x)
    p, v, pl = networkobject.simulate()

    # Penalty for not supplying the required minimum pressure at each node
    pH = 0

    SOURCE_NODES = 1
    for i in range(len(p) - SOURCE_NODES):
        if p[i] < _hmin or p[i] < 0:
            pH += _penalty

    # Penalty for violating the flow velocity limits of the pipes
    pV = 0
    for i in range(len(v)):
        if v[i] < _vmin or v[i] > _vmax:
            pV += _penalty

    # Objective function
    cost = 0
    for i in range(len(x)):
        c = unit_prices[x[i]] * pl[i] * (pV+1) * (pH+1)
        cost += c
    return cost


def network_size(networkobject, x, **kwargs):
    """" The objective function for dimensioning optimization

    :param networkobject, an implementation of a 'Network' class
    :param x, a list of pipe diameters
    :return fitness, minimum is better
    """
    unit_prices = kwargs.get('price', defaults)
    _hmin = kwargs.get('hmin', 30.0)
    _hmax = kwargs.get('hmax', 100.0)
    _vmin = kwargs.get('vmin', 0.50)
    _vmax = kwargs.get('vmax', 2.50)
    _penalty = kwargs.get('penalty_step', 1)

    # Perform the EPANET simulation
    networkobject.change_diameters(x)
    p, v, pl = networkobject.simulate()

    # Penalty for not supplying the required minimum pressure at each node
    pH = 0

    SOURCE_NODES = 1
    for i in range(len(p) - SOURCE_NODES):
        if p[i] < _hmin:
            pH += _penalty

    # Penalty for violating the flow velocity limits of the pipes
    pV = 0
    for i in range(len(v)):
        if v[i] < _vmin or v[i] > _vmax:
            pV += _penalty

    # Objective function
    cost = 0
    for i in range(len(x)):
        c = unit_prices[x[i]] * pl[i] * (pV+1) * (pH+1)
        cost += c
    return cost


def network_layout(x, coordx, coordy, coordz, **kwargs):
    """ Objective function for network layout

    Sums the distance of all the connections or links in the network's
    nodes.

    :param x, list of connections or links in the network (Genome instance)
    :param coords,
    :
    :return dist, total distance of the network (less is better)
    """

    # Compute distance between the two points, using 3D coordinates (x, y, z)
    penalty = 0
    dist = 0
    for i in range(len(x)):
        # This works because "x" is lower than "coordx", "coordy" and "coordz"
        # by one element
        x1, y1, z1 = coordx[i+1], coordy[i+1], coordz[i+1]
        # print x1, y1, z1
        x2, y2, z2 = coordx[x[i]], coordy[x[i]], coordz[x[i]]
        dist += sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # Penalty if the node connects with itself (position equal to value)
        if x[i] is i+1:
            penalty += 1

    # Penalties, this is included to avoid loops and splitted networks, also
    # works for duplicated connections (two points connected with each other)
    # Idea taken from Ponce, (2013):
    #    "Supply order" is an index assigned to each node according to how far
    #    is connected from the source node. The source is order 0, all the
    #    nodes connected directly to source are order 1, order 2 nodes are
    #    connected to order 1 nodes, and so on.
    l = len(x)
    supplyOrder = [0] * l
    supplyOrder[0] = 1
    for i in range(1, l+1):
        for j in range(1, l+1):
            if supplyOrder[j-1] == i:
                for k in range(1, l+1):
                    if x[k-1] == j:
                        supplyOrder[k-1] = supplyOrder[j-1] + 1
    # print(supplyOrder)

    # Count penalties related to supply order
    for i in range(l):
        if supplyOrder[i] == 0:
            penalty += 5
    # Penalty if the genome does not contain at least one zero element
    if 0 not in x:
        # Increasing by 1 seems to have no effect, trying with a higher value
        penalty += 10

    # Be sure penalty is not zero, because it's multiplicative
    if penalty == 0:
        penalty = 1
    return dist * (penalty)


def length_layout(tree, x, y, base, *args):
    """ Length of network layout

    :param tree, list of arcs in the network (Genome instance)
    :param x, list with x-coordinates
    :param y, list with y-coordinates
    :return length, total length of the network
    """
    penalty = 1
    length = 0
    for arc in tree:
        # Increase penalty if an arc is repeated
        occurrences = tree.getInternalList().count(arc)
        if occurrences > 1:
            penalty += occurrences
        # Get the nodes from the arc
        n = get_nodes(base[arc])
        # Get the coordinates of the arc nodes
        x1, y1 = x[n[0]], y[n[0]]
        x2, y2 = x[n[1]], y[n[1]]
        # Calculate the distance between the nodesof the arc
        length += sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Convert integer representation to string
    tree_str = ind2arc(tree, base)

    # Check for supply order
    so, p = supply_order(tree_str, len(x))
    penalty += p
    return length * penalty


def supply_order(arcs, nodes):
    """ Supply order

    Defines the supply order from a list of nodes

    :param arcs, a list of arcs forming a network
    :param nodes, an integer number of nodes
    """
    penalty = 0
    node = 0     # Root node
    memory = []  # Nodes that already have an order
    memory.append(node)
    supplyOrder = []

    # Initialize supply order list with zeros
    for i in range(nodes):
        supplyOrder.append(0)

    for n in range(nodes):
        adj = adjacent(n, arcs)
        if len(adj) is 0:
            # If no adjacent arcs, then penalty
            # print("Node {0} has no adjacent arcs!".format(n))
            penalty += 1
        else:
            # There is at least one adjacent arc to the node
            for j in adj:
                arc = arcs[j]
                # Get the nodes of the current arc
                anodes = get_nodes(arc)

                if (anodes[0] in memory) and (anodes[1] not in memory):
                    # print("{0} exist, adding {1}".format(anodes[0],
                    #        anodes[1]))
                    supplyOrder[anodes[1]] = supplyOrder[anodes[0]] + 1
                    memory.append(anodes[1])
                    # print(memory)
                elif (anodes[0] not in memory) and (anodes[1] in memory):
                    # print("{0} exist, adding {1}".format(anodes[1],
                    #        anodes[0]))
                    supplyOrder[anodes[0]] = supplyOrder[anodes[1]] + 1
                    memory.append(anodes[0])
                    # print(memory)
                elif (anodes[0] not in memory) and (anodes[1] not in memory):
                    # print("{0} and {1} not exist".format(anodes[0],
                    #        anodes[1]))
                    supplyOrder[anodes[0]] = 0
                    supplyOrder[anodes[1]] = supplyOrder[anodes[0]] + 1
                    memory.append(anodes[0])
                    memory.append(anodes[1])
                    # print(memory)
                    penalty += 1
    # print("Total penalties: {0}".format(penalty))
    return supplyOrder, penalty


def unit_cost(unit_prices, diameter):
    """ Unit cost

    Returns the cost per unit of length for pipe of the specified diameter.

    Cost data obtained from:
      Xiaojun Zhou and David Y. Gao and Angus R. Simpson, 2016
      Optimal design of water distribution networks by a discrete state
      transition algorithm, Engineering Optimization Vol. 48, Num. 4, 603-628

    :param int diameter, index of the diameter in the list of available ones
    """
    assert type(diameter) is int, 'Required integer type'
    assert type(unit_prices) is dict, 'Unit prices should be dictionary'
    return unit_prices[diameter]


def init_tree(A, x, y, base, *args):
    """ Tree Growing Algorithm

    Algorithm from Walters & Smith (1995)

    :param A, list of arcs within the growing tree
    :param x, list with X coordinates
    :param y, list with Y coordindates
    :param base, base graph of connections among nodes
    """
    total_nodes = len(x)
    C = []   # set of nodes contained within the growing tree
    # A = []   # set of arcs within the growing tree
    AA = []  # set of arcs adjacent to the growing tree

    # 1) Identify the root node Nr
    # Nr = 0
    Nr = np.random.randint(total_nodes)

    # 2) Initialise C = [Nr]
    C.append(Nr)

    # 3) Initialise A = [ ]
    # 4) Initialise AA = [arcs in base graph connected to root'node]
    AA = adjacent(Nr, base)

    iters = 0
    while len(A) != (total_nodes - 1):
        # 5) Choose arc, a, at random from AA
        # probabilities = prob(AA)
        # print('  Probabilities={0}'.format(probabilities))
        # a = int(np.random.choice(AA, 1, p=probabilities))
        a = int(np.random.choice(AA, 1))

        # 6) A = A + [a]
        A.append(a)

        # 7) Identify newly connected node, N
        # 8) C = C + [N]
        nodes = get_nodes(base[a])
        N = -1  # Just initialize with something dummy
        for node in nodes:
            if node not in C:
                N = node
                C.append(N)
        # 9) Identify arcs, ac(i), connected to N in base graph,
        #    (excluding arc a)
        aci = adjacent(N, base)
        if a in aci:
            aci.remove(a)

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
                if nds[0] in C and nds[1] in C:
                    # Do nothing
                    dummy = 1
                else:
                    # Assuming correct direction
                    AA.append(ac)
        iters += 1
    # Print population
#    for individual in A:
#        print(A.getInternalList())
#    return A
