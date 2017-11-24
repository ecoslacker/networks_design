# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:26:29 2017

@author: ecoslacker
"""
import os
from datetime import datetime
from pyevolve import G1DList, Consts, GSimpleGA, DBAdapters, Selectors, \
     Mutators, Crossovers, Scaling
from objectivefunctions import network_size, init_diameters, init_layout, \
     network_layout, network_size_cost
from hydraulic import Network
from graphics import plot_convergence, plot_layout_network, plot_nodes
from utilities import read_instance

diameters = {25: 2,
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

# pipe_prices = {125: 58,
#               150: 62,
#               200: 71.2,
#               250: 88.9,
#               300: 112.3,
#               350: 138.7,
#               400: 169,
#               450: 207,
#               500: 248,
#               550: 297,
#               600: 347,
#               650: 405,
#               700: 470}

pipe_prices = {80: 23,
               100: 32,
               120: 50,
               140: 60,
               160: 90,
               180: 130,
               200: 170,
               220: 300,
               240: 340,
               260: 390,
               280: 430,
               300: 470,
               320: 500}


class OptimizationEngine:
    """ Optimization Engine

    The engine provides an abstraction class for the optimal design of water
    distribution systems. Only irrigation (tree-like) networks are considered.
    The available metaheuristics include GAs and ACO.

    :param str problem: the problem to solve 'LAYOUT', 'DIMEN' or 'BOTH'
    :param str instance: path and file name of the instance data
    :param str metaheuristic: the method to solve the instance 'GA' or 'ACO'
    """
    def __init__(self, problem, instance, metaheuristic, **kwargs):
        """ Initialization function
        """

        # Default pipe prices from Zhou (2015)
        default_prices = {25: 2,
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
        self.problems = ['LAYOUT', 'DIMEN', 'BOTH']
        self.metaheuristics = ['GA', 'ACO']

        # Check
        assert problem in self.problems, "Invalid problem"
        assert metaheuristic in self.metaheuristics, "Invalid metaheuristic"
        assert os.path.exists(instance) is True, "Instance file data not found"

        # Default arguments
        self.pipe_prices = kwargs.get('pipe_prices', default_prices)

        return


def ga_dimensions(dim, objfunc, initfunc, statsfile, network, **kwargs):
    """ Run Genetic Algorithm

    Runs the Genetic Algorithm using the Pyevolve module, an implementation of
    the Simple GA is used to minimize the objective function.

    :param int dim, dimension of the problem (number of pipes of the network)
    :param callable objfunc, objective function object
    :param callable initfunc, initialize function object
    :param str statsfile, the file name of a CSV tect file to save the results
    :param Network network, an object which diameters will be optimized
    :param int run, the number of the current execution
    :param int gen, generations of the GA
    :param int pop, population size of the GA
    :param float cross, crossover rate of the GA
    :param float mut, mutation rate of the GA
    :return G1DList best, a list of the diameters with the maximum fitness
    """
    # Get the arguments
    run = kwargs.get('run', 0)
    gen = kwargs.get('gen', 100)
    pop = kwargs.get('pop', 80)
    xover = kwargs.get('xover', 0.9)
    mut = kwargs.get('mut', 0.02)

    # Genome instance, set the dimension and initialization function
    genome = G1DList.G1DList(dim)
    genome.initializator.set(lambda x, **kwargs: initfunc(x, dim,
                                                          price=pipe_prices))

    # The evaluator function (objective function)
    genome.evaluator.set(lambda x, **kwargs: objfunc(network, x,
                                                     price=pipe_prices))
    genome.mutator.set(Mutators.G1DListMutatorSwap)
    genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)

    # Genetic Algorithm instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMinimax(Consts.minimaxType["minimize"])

    # Use roulette wheel with linear scaling
    p = ga.getPopulation()
    p.scaleMethod.set(Scaling.LinearScaling)

    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(gen)
    ga.setPopulationSize(pop)
    ga.setCrossoverRate(xover)
    ga.setMutationRate(mut)

    # Save stats to CSV file for later analysis
    csv_adapter = DBAdapters.DBFileCSV(identify='run' + str(run),
                                       filename=statsfile,
                                       reset=False)
    ga.setDBAdapter(csv_adapter)

    # Do the evolution, with stats dump
    # frequency of certain generations
    ga.evolve(freq_stats=25)
    stats = ga.getStatistics()
    print(stats)

    best = ga.bestIndividual()
    p = ga.getPopulation()
    return best, p


def optim_dimensions_ga(path, inputfile, executions, **kwargs):
    """ Optimize dimensios with GA

    Optimization of the dimensions (diameters) of the specified network using
    an implementation of a Simple Genetic Algorithm

    :param str path, the working directory path
    :param str inputfile, the name of the *.inp file (EPANET's input text file)
    :param int executions, number of executions of the SGA
    :param int gen, GA generations
    :param int pop, GA population size
    :param float xover, GA crossover rate
    :param float mut, GA mutation rate
    :param str save_dir, directory to save the GA executions data
    :param callable obj_func, the objective function name (NOT a string!)
    :param callable init_func, the initialization function name
    :return best, best individual found in the GA executions
    """

    start = datetime.now()

    # Get the arguments
    _gen = kwargs.get('gen', 100)
    _pop = kwargs.get('pop', 80)
    _xover = kwargs.get('xover', 0.9)
    _mut = kwargs.get('mut', 0.02)
    save_dir = kwargs.get('save_dir', 'results_ga/')
    objfunc = kwargs.get('obj_func', network_size)
    initfunc = kwargs.get('init_func', init_diameters)

    # Create the Network object needed for EPANET simulation and analysis
    mynet = Network(path, inputfile)
    mynet.open_network()
    mynet.initialize()

    # Remember the best individual of all runs
    best = None
    f = '%Y_%m_%d-%H_%M_%S'
    stats = save_dir + 'ga_dimen_' + datetime.strftime(start, f) + '.csv'
    sol_file = stats[:-4] + '.txt'

    # lbest = []
    for i in range(executions):
        print('Execution {0}'.format(i))
        print('-' * 80)

        # Get the best result of the current run
        b, pot = ga_dimensions(mynet.pipes, objfunc, initfunc, stats, mynet,
                               run=i, gen=_gen, pop=_pop, xover=_xover,
                               mut=_mut)

        # Initialize the best individual
        if (i is 0) or (b.getRawScore() < best.getRawScore()):
            best = b
        # lbest.append(best.getRawScore())

    # Save the diameters found in a INP file and close the Network object
    mynet.change_diameters(best.genomeList)
    mynet.save_inp_file(stats[:-4] + '_solution.inp')
    mynet.close_network()

    # Save best solution to a text file
    with open(sol_file, 'w') as f:
        f.write('{0}'.format(best))

    plot_convergence(executions, _gen, stats)

    # Runtime
    runtime = datetime.now() - start
    print('Run time: {0}'.format(runtime))
    with open(sol_file, 'a') as f:
        f.write('Run time: {0}'.format(runtime))

    return best, pot


def ga_layout(dim, objfunc, initfunc, statsfile, coordx, coordy, coordz,
              **kwargs):
    """ Run Genetic Algorithm

    Runs the Genetic Algorithm using the Pyevolve module, an implementation of
    the Simple GA is used to minimize the objective function.

    :param dim, dimension of the problem (number of nodes of the network)
    :param objfunc, objective function object
    :param initfunc, initialize function object
    :param statsfile, the file name of a CSV tect file to save the results
    :param coordx, list with x-axis coordinates
    :param coordy, list with y-axis coordinates
    :param run, the number of the current execution
    :param gen, generations of the GA
    :param pop, population size of the GA
    :param cross, crossover rate of the GA
    :param mut, mutation rate of the GA
    :return best, a list of the diameters with the maximum fitness
    """
    # Get the arguments
    run = kwargs.get('run', 0)
    gen = kwargs.get('gen', 100)
    pop = kwargs.get('pop', 80)
    xover = kwargs.get('xover', 0.9)
    mut = kwargs.get('mut', 0.02)

    # Genome instance
    genome = G1DList.G1DList(dim)
    genome.initializator.set(lambda x, **args: initfunc(x, dim))

    # The evaluator function (objective function)
    genome.evaluator.set(lambda x, **args: objfunc(x, coordx, coordy, coordz))

    genome.mutator.set(Mutators.G1DListMutatorSwap)
    genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)

    # Genetic Algorithm instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMinimax(Consts.minimaxType["minimize"])

    # Use roulette wheel with linear scaling
    p = ga.getPopulation()
    p.scaleMethod.set(Scaling.LinearScaling)

    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(gen)
    ga.setPopulationSize(pop)
    ga.setCrossoverRate(xover)
    ga.setMutationRate(mut)

    # Save stats to CSV file for later analysis
    csv_adapter = DBAdapters.DBFileCSV(identify="run" + str(run),
                                       filename=statsfile,
                                       reset=False)
    ga.setDBAdapter(csv_adapter)

    # Do the evolution, with stats dump
    # frequency of certain generations
    ga.evolve(freq_stats=100)
    stats = ga.getStatistics()
    print(stats)

    best = ga.bestIndividual()
    return best


def optim_layout_ga(instance, executions, **kwargs):
    """ Network layout optimization

    :param str instance, problem instance or file name
    :param int executions, times the GA will be executed
    :param int gen, GA generations
    :param int pop, GA population size
    :param float cross, GA crossover rate
    :param float mut, GA mutation rate
    :param str save_dir, directory to save the GA executions data
    :param bool base_graph, True to use data from a base graph file
    :param callable obj_func, the objective function name (NOT a string!)
    :param callable init_func, the initialization function name
    :return best, best individual found in the GA executions
    """

    start = datetime.now()

    # Get the arguments
    _gen = kwargs.get('gen', 100)
    _pop = kwargs.get('pop', 80)
    _xover = kwargs.get('xover', 0.9)
    _mut = kwargs.get('mut', 0.02)
    save_dir = kwargs.get('save_dir', 'results_ga/layout_runs/')
    base_graph = kwargs.get('base_graph', False)
    objfunc = kwargs.get('obj_func', network_layout)
    initfunc = kwargs.get('init_func', init_layout)

    # Remember the best individual of all runs
    best = None
    cx, cy, cz, name, _labels, base = read_instance(instance, base_graph)
    dim = len(cx) - 1         # Problem dimension is pipe number
    f = '%Y_%m_%d-%H_%M_%S'   # Date format
    stats = save_dir + 'ga_layout_' + datetime.strftime(start, f) + '.csv'
    net_img = stats[:-4] + '_network.png'
    sol_file = stats[:-4] + '.txt'

    # Execute the GA the specified times
    for i in range(executions):
        print('Execution {0}'.format(i))
        print('-' * 80)

        # Get the best result of the current run
        b = ga_layout(dim, objfunc, initfunc, stats, cx, cy, cz, run=i,
                      gen=_gen, pop=_pop, xover=_xover, mut=_mut)

        # Initialize the best individual
        if (i is 0) or (b.getRawScore() < best.getRawScore()):
            best = b

    # Save best solution to a text file
    with open(sol_file, 'w') as f:
        f.write('{0}'.format(best))

    # Plot convergence
    plot_convergence(executions, _gen, stats)

    # Plot network
    plot_layout_network(cx, cy, best, executions, net_img, labels=_labels)

    # Runtime
    runtime = datetime.now() - start
    print('Run time: {0}'.format(runtime))
    with open(sol_file, 'a') as f:
        f.write('Run time: {0}'.format(runtime))
    return best

# *** END OF THE CLASSES AND FUNCTIONS ***

if __name__ == "__main__":

    start_time = datetime.now()

    # *** TEST CODE FOR LAYOUT OPTIMIZATION ***
    execs = 20

    # 9 nodes network from Geem et al. (2000)
#    instance = 'data/network09.csv'
#    directory = 'results_ga/layout_network09/'

    # 12 nodes network from Angeles-Montiel (2002)
#    instance = 'data/network12.csv'
#    directory = 'results_ga/layout_network12/'

    # 64 nodes network from Walters & Lohbeck (1993)
#    instance = 'data/network64.csv'
#    directory = 'results_ga/layout_network64/'

    # Run optimization
    #plot_nodes(instance, False)
#    best = optim_layout_ga(instance, execs, pop=200, gen=500, xover=0.85,
#                           mut=0.1, save_dir=directory)

    # ******** OPTIMIZATION OF NETWORK DIMENSIONS (DIAMETERS) ********
#    path = 'data/'
#    inp_file = 'TwoLoop.inp'
#    directory = 'results_ga/dimen_twoloop/'

    path = 'solutions/'
    inp_file = 'network09_sol_afshar2008_2.inp'
    directory = 'results_ga/dimen_network09/'

    best, pop = optim_dimensions_ga(path, inp_file, execs, pop=200, gen=500,
                                    xover=0.85, mut=0.05, save_dir=directory)

    # Print best
    print('**** OVERALL BEST ****')
    print(best)

    run_time = datetime.now() - start_time
    print('\n{0} executions completed in: {1}'.format(execs, run_time))
