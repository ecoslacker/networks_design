# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:41:22 2017

@author: ecoslacker
"""

# import matplotlib
# matplotlib.use('TkAgg')

import csv

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.path import Path
from utilities import read_instance


def plot_layout_network(cx, cy, c, run=1, filename='', labels=[]):
    """ Plot layout network

    Plots the nodes and connections of a network
    :param cx, a list with X-Coordinates
    :param cy, a list with Y-Coordinates
    :param c, a list of integers with the connections among nodes
    """
    assert len(cx) == len(cy), 'Legnth of coordinates list does not match'
    lo = 8  # label offset
    if len(labels) == 0:
        labels = ['{0}'.format(i) for i in range(len(cx))]
    assert len(labels) == len(cx), 'Length of labels list does not match'

    figNetworkLayout = plt.figure()
    ax = figNetworkLayout.add_subplot(1, 1, 1)

    plt.scatter(cx, cy, c='black')
    # Plot labels
    for label, x, y in zip(labels, cx, cy):
        plt.annotate(label, xy=(x+lo, y+lo), xytext=(1, 1),
                     textcoords='offset points')
    # Draw the connections
    for i in range(len(c)):
        x1, y1 = cx[i+1], cy[i+1]
        x2, y2 = cx[c[i]], cy[c[i]]
        # Draw a line from (x1, y1) to (x2, y2)
        verts = [(x1, y1), (x2, y2)]
        codes = [Path.MOVETO, Path.LINETO]
        path = Path(verts, codes)
        ax.add_patch(patches.PathPatch(path, color='black', lw=0.5))

    plt.title('Best network of: ' + str(run) + ' executions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.ylim([min(cy)-50, max(cy)+65])
    plt.xlim([min(cx)-50, max(cx)+50])
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    figNetworkLayout.show()


#def plot_chart(file_stats, run=0):
#    """ Creates a convergence chart
#    Plot the fitness of each generation to create a convergence chart, using
#    the data saved in the file created by the evolution.
#
#    :param file_stats, CSV file containing data from evolution
#    :param run, number of the current execution or run
#    """
#
#    # Read the data from CSV file, the format is as follow
#    # column 0: identify
#    # column 1: generation
#    # column 2: rawMin
#    # column 3: fitAve
#    # column 4: fitMin
#    # column 5: rawVar
#    # column 6: rawDev
#    # column 7: rawAve
#    # column 8: fitMax
#    # column 9: rawMax
#
#    gen = []
#    fit = []
#    with open(file_stats, 'r') as f:
#        reader = csv.reader(f, delimiter=';')
#        for row in reader:
#            gen.append(int(row[1]))    # Generation
#            fit.append(float(row[3]))  # Fitness average
#
#    # Plot the data: generation vs fitness
#    figConv = plt.figure()
#    plt.plot(gen, fit)
#    plt.plot(gen, fit)
#    plt.title('Convergence chart for run: {0}'.format(run))
#    plt.xlabel('Generation')
#    plt.ylabel('Fitness (Average)')
#    figConv.show()


def plot_maxmin_conv(runs, gens, file_stats):
    """ Creates a convergence chart
    Plot the fitness of each generation to create a convergence chart, using
    the data saved in the file created by the evolution.

    :param runs, number of executions or runs
    :param gens, generations of each execution
    :param file_stats, CSV file containing data from evolution
    """

    # Read the data from CSV file, the format is as follow
    # column 0: identify
    # column 1: generation
    # column 2: rawMin
    # column 3: fitAve
    # column 4: fitMin
    # column 5: rawVar
    # column 6: rawDev
    # column 7: rawAve
    # column 8: fitMax
    # column 9: rawMax

    # Create the figure
    figConvergence = plt.figure()

    # Read all the data from the file
    gen = []
#    fit = []
#    fmin = []
#    fmax = []
    rmin = []
#    rave = []
#    rmax = []
    with open(file_stats, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            gen.append(int(row[1]))     # Generation
#            fit.append(float(row[3]))   # Fitness average
#            fmin.append(float(row[4]))  # Fitness minimum
#            fmax.append(float(row[8]))  # Fitness maximum
            rmin.append(float(row[2]))   # Raw minimum
#            rave.append(float(row[2]))
#            rmax.append(float(row[2]))

    # Get the data slicing through executions, each run contains all its
    # generations and fitness
    ini = 0
    end = gens
    best = rmin[ini:end]
    g = gen[ini:end]
    for r in range(runs):
        # g = gen[ini:end]
        y = rmin[ini:end]
        for i in range(len(y)):
            if y[i] < best[i]:
                best[i] = y[i]
#        y1 = rave[ini:end]
#        y2 = rmax[ini:end]
        # Plot the data: generation vs fitness
#        plt.plot(g, y)
#        plt.plot(g, y1)
#        plt.plot(g, y2)
        ini += gens
        end += gens

    plt.plot(g, best, c='black')

    # Create a new file with PNG extension to save the chart
    file_fig = file_stats[:-4]
    file_fig = file_fig + ".png"

    plt.title('Convergence chart for {0} executions.'.format(runs))
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figConvergence.show()


def plot_convergence(runs, gens, file_stats):
    """ Creates a convergence chart

    Plot the fitness of each generation to create a convergence chart, using
    the data saved in the file created by the evolution.

    :param runs, number of executions or runs
    :param gens, generations of each execution
    :param file_stats, CSV file containing data from evolution
    """

    # Read the data from CSV file, the format is as follow
    # column 0: identify
    # column 1: generation
    # column 2: rawMin
    # column 3: fitAve
    # column 4: fitMin
    # column 5: rawVar
    # column 6: rawDev
    # column 7: rawAve
    # column 8: fitMax
    # column 9: rawMax
    COL_GEN = 1
    COL_MIN = 2

    # Create the figure
    figConvergence = plt.figure()

    # Read all the data from the file
    gen = []
    rmin = []

    with open(file_stats, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            gen.append(int(row[COL_GEN]))      # Generation
            rmin.append(float(row[COL_MIN]))   # Raw minimum

    # Get the data slicing through executions, each run contains all its
    # generations and fitness
    ini = 0
    end = gens
    best = rmin[ini:end]
    g = gen[ini:end]

    # Get the best of each generation
    for r in range(runs):
        y = rmin[ini:end]
        for i in range(len(y)):
            if y[i] < best[i]:
                best[i] = y[i]
        ini += gens
        end += gens

    # Plot the data: generation vs best fitness
    plt.plot(g, best, c='black')

    # Create a new file with PNG extension to save the chart
    file_fig = file_stats[:-4]
    file_fig = file_fig + ".png"

    plt.title('Convergence chart for {0} executions.'.format(runs))
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figConvergence.show()


def plot_conv_graph(execs, iters, file_stats):
    """ Creates a convergence chart

    Plot the fitness of each iteration to create a convergence chart, using
    the data saved in the file created by the evolution.

    :param runs, number of executions or runs
    :param gens, generations of each execution
    :param file_stats, CSV file containing data from evolution
    """

    COL_GEN = 1
    COL_MIN = 2

    # Create the figure
    figConvergence = plt.figure()

    # Read all the data from the file
    gen = []
    rmin = []

    with open(file_stats, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            gen.append(int(row[COL_GEN]))      # Generation
            rmin.append(float(row[COL_MIN]))   # Raw minimum

    # Get the data slicing through executions, each run contains all its
    # iterations and fitness
    ini = 0
    end = iters
    best = rmin[ini:end]
    it = gen[ini:end]

    # Get the best of each iteration
    for r in range(execs):
        y = rmin[ini:end]
        for i in range(len(y)):
            if y[i] < best[i]:
                best[i] = y[i]
        ini += iters
        end += iters

    # Plot the data: generation vs best fitness
    plt.plot(it, best, c='black')

    # Create a new file with PNG extension to save the chart
    file_fig = file_stats[:-4]
    file_fig = file_fig + ".png"

    plt.title('Convergence chart for {0} executions.'.format(execs))
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figConvergence.show()


def plot_conv(raw, execs):
    """ Creates a convergence chart
    Plot the raw minimum values of each generation to create a
    convergence chart.

    :param gen, list with generations
    :param raw, list of raw value of best individual
    :param execs, number of executions
    """

    # Plot the data: generation vs raw value
    figConv2 = plt.figure(1)
    plt.plot(raw)
    plt.title('Convergence chart for: {0} executions'.format(execs))
    plt.xlabel('Executions')
    plt.ylabel('Cost')
    figConv2.show()


#def plot_convergence_list(x, y, execs, figure):
#    """ Creates a convergence chart
#
#    Plot data from two lits.
#
#    :param list x, list with iterations
#    :param list y, list with cost values for the best individual
#    :param int execs, number of executions
#    :param str figure, file name to save the figure
#    """
#    figConvList = plt.figure(7)
#    plt.plot(x, y, c='black')
#    plt.title('Convergence chart for: {0} executions'.format(execs))
#    plt.xlabel('Iteration')
#    plt.ylabel('Cost')
#    plt.savefig(figure, bbox_inches='tight', dpi=300, transparent=True)
#    figConvList.show()


def plot_nodes(filename, base=False):
    """ Plot nodes of a network

    Open a file with (x,y) coordinates and plot the nodes of the network, if
    base graph is specified then draws the available links.

    :param filename, the text file with the node's coordinates
    :param base, True to plot the base graph of the network
    """

    x, y, z, name, labels, base_graph = read_instance(filename, base)

    # Plot the nodes of the network
    figNodes = plt.figure()
    ax = figNodes.add_subplot(1, 1, 1)

    plt.scatter(x, y, c='black')
    for label, lx, ly in zip(labels, x, y):
        plt.annotate(label, xy=(lx, ly), xytext=(3, 2),
                     textcoords='offset points')
    if base:
        for link in base_graph:
            # print("Drawing link: {0}".format(link))
            n1 = int(link.split('-')[0])
            n2 = int(link.split('-')[1])
            x1, y1 = x[n1], y[n1]
            x2, y2 = x[n2], y[n2]
            # Draw a line from (x1, y1) to (x2, y2)
            verts = [(x1, y1), (x2, y2)]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            ax.add_patch(patches.PathPatch(path, color='black', lw=0.5))
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
#    plt.ylim([min(y)-50, max(y)+50])
#    plt.xlim([min(x)-50, max(x)+50])
    file_fig = filename[:-4]
    file_fig = file_fig + ".png"
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figNodes.show()


def plot_stylized_network(filename, base=False):
    """ Plot a stylized network

    Open a file with (x,y) coordinates and plot the nodes of the network, if
    base graph is specified then draws the available links.

    :param filename, the text file with the node's coordinates
    :param base, True to plot the base graph of the network
    """
    x, y, z, name, labels, base_graph = read_instance(filename, base)

    # Plot the nodes of the network
    figStyle = plt.figure(6)
    ax = figStyle.add_subplot(1, 1, 1)

    plt.scatter(x, y, s=300, c='black')
    for label, lx, ly in zip(labels, x, y):
        plt.annotate(label, xy=(lx, ly), xytext=(-5, -3),
                     textcoords='offset points', color='white')

    if base:
        for link in base_graph:
            print("Drawing link: {0}".format(link))
            n1 = int(link.split('-')[0])
            n2 = int(link.split('-')[1])
            x1, y1 = x[n1], y[n1]
            x2, y2 = x[n2], y[n2]
            # Draw a line from (x1, y1) to (x2, y2)
            verts = [(x1, y1), (x2, y2)]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            ax.add_patch(patches.PathPatch(path, color='black', lw=0.5))

    plt.title(name)
    plt.axis('equal')
    # Create a new file with PNG extension to save the chart
    file_fig = filename[:-4]
    file_fig = file_fig + ".png"
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figStyle.show()

if __name__ == '__main__':
    network = 'data/network12.csv'
    # plot_stylized_network(network, False)
    # plot_nodes(network)
    plot_stylized_network(network, True)
