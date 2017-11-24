#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tree-like networks layout (topology) optimization using ACO metaheuristic

Created on Tue Sep 26 13:38:06 2017
@author: ecoslacker
"""
from datetime import datetime
from graphics import plot_conv_graph
from yaaco import ACO, Ant, OptimizeDimensionsACO


def optim_layout_aco(instance, executions, **kwargs):
    """ Optimize layout with ACO

    Optimization of tree-like networks layout (topology) using ACO

    :param instance: file name of the problem instance
    :param int executions: times the ACO algorithm will be executed
    :param str savedir: path to save the results and statistics
    :param int n_ants: number of ants in the colony
    :param float rho: the pheromone evaporation parameter
    :param float alpha: the pheromone trail influence
    :param float beta: the heuristic information influence
    :param int max_iters: maximum number of iterations of the algorithm
    :param str ptype: Description of problem instance (Default 'TREE_NET')
    :param bool base: True to use data from a base graph file
    """
    # Get the parameters
    savedir = kwargs.get('savedir', 'results_aco/')
    n_ants = kwargs.get('n_ants', 10)
    _rho = kwargs.get('rho', 0.02)
    _alpha = kwargs.get('alpha', 1.0)
    _beta = kwargs.get('beta', 2.0)
    _iters = kwargs.get('max_iters', 20)
    _ptype = kwargs.get('ptype', 'TREE_NET')
    _base = kwargs.get('base', False)

    best = Ant(1, _ptype)
    t = datetime.now()       # File names will be identified by its exec time
    f = '%Y_%m_%d-%H_%M_%S'  # Date format
    # File names to save the statistics, network plot and solution
    _stats = savedir + 'aco_layout_' + datetime.strftime(t, f) + '.csv'
    network_img = _stats[:-4] + '_network.png'
    base_img = _stats[:-4] + '_base.png'
    sol_file = _stats[:-4] + '.txt'

    # Execute the ACO algorithm the specified times
    for i in range(executions):
        print('Execution {0}'.format(i))
        print('-' * 80)

        # Get the best solution from the current execution
        aco = ACO(n_ants, instance, rho=_rho, alpha=_alpha, beta=_beta,
                  max_iters=_iters, use_base_graph=_base,
                  instance_type=_ptype, stats=_stats)
        b = aco.run(identify=i)

        # Initialize the best individual
        if (i == 0) or (b.tour_length < best.tour_length):
            best = b

    # Print best ant (solution)
    print('*** OVERALL BEST ***')
    print(best)

    # Plot the base graph & best solution
    aco.plot_base_graph(base_img)
    aco.plot_best(network_img, use_this_ant=b)
    plot_conv_graph(executions, _iters, _stats)

    # Save the best solution to a text file
    with open(sol_file, 'w') as f:
        f.write('{0}'.format(aco))
        f.write('-' * 80)
        f.write('\n*** OVERALL BEST ***\n')
        f.write('{0}'.format(best))
        f.write('Runtime: {0}'.format(aco.exec_time))

    return best


def optim_dimen_aco(instance, executions, **kwargs):
    """ Optimize dimensions with ACO

    Optimization of pipe network dimensions (diameters) using ACO

    :param instance: file name of the problem instance
    :param int executions: times the ACO algorithm will be executed
    :param str savedir: path to save the results and statistics
    :param int n_ants: number of ants in the colony
    :param float rho: the pheromone evaporation parameter
    :param float alpha: the pheromone trail influence
    :param float beta: the heuristic information influence
    :param int max_iters: maximum number of iterations of the algorithm
    :param str ptype: Description of problem instance (Default 'DIMEN_NET')
    :param bool base: True to use data from a base graph file
    """
    # Get the parameters
    savedir = kwargs.get('savedir', 'results_aco/')
    n_ants = kwargs.get('n_ants', 10)
    _rho = kwargs.get('rho', 0.02)
    _alpha = kwargs.get('alpha', 1.0)
    _beta = kwargs.get('beta', 2.0)
    _iters = kwargs.get('max_iters', 20)
    _ptype = kwargs.get('ptype', 'DIMEN_NET')
    # _base = kwargs.get('base', False)

    best = Ant(1, _ptype)
    t = datetime.now()       # File names will be identified by its exec time
    f = '%Y_%m_%d-%H_%M_%S'  # Date format
    # File names to save the statistics, plot and solution
    _stats = savedir + 'aco_layout_' + datetime.strftime(t, f) + '.csv'
    sol_file = _stats[:-4] + '.txt'

    # Execute the ACO algorithm the specified times
    for i in range(executions):
        print('Execution {0}'.format(i))
        print('-' * 80)

        # Get the best solution from the current execution
        aco = OptimizeDimensionsACO(n_ants, instance, rho=_rho, alpha=_alpha,
                                    beta=_beta, max_iters=_iters)
        b = aco.run(identify=i)

        # Initialize the best individual
        if (i == 0) or (b.tour_length < best.tour_length):
            best = b

    # Print best ant (solution)
    print('*** OVERALL BEST ***')
    print(best)

    # Plot the convergence graph & best solution
    plot_conv_graph(executions, _iters, _stats)

    # Save the best solution to a text file
    with open(sol_file, 'w') as f:
        f.write('{0}'.format(aco))
        f.write('-' * 80)
        f.write('\n*** OVERALL BEST ***\n')
        f.write('{0}'.format(best))
        f.write('Runtime: {0}'.format(aco.exec_time))

    return best

if __name__ == "__main__":

    start_time = datetime.now()
    execs = 20

    # *** TEST CODE FOR OPTIMIZATION OF NETWORK DIMENSIONS (DIAMETERS) ***
    path = 'solutions/'
    file_name = 'network09_sol_afshar2008_2.inp'
    optim_dimen_aco()

    # *** TEST CODE FOR LAYOUT OPTIMIZATION ***
    # 9 nodes network from Geem et al. (2000)
#    instance = 'data/network09.csv'
#    save_dir = 'results_aco/layout_network09/'

    # 12 nodes network from Angeles-Montiel (2002)
#    instance = 'data/network12.csv'
#    save_dir = 'results_aco/layout_network12/'

    # 64 nodes network from Walters & Lohbeck (1993)
#    instance = 'data/network64.csv'
#    directory = 'results_ga/layout_network64/'

    # Parameters for Ant System
#    m = 5
#    r = 0.5
#    aco_iters = 10
#    problem = 'TREE_NET'


    # Run the layout optimization
#    best = optim_layout_aco(instance, execs, n_ants=m, rho=r,
#                            max_iters=aco_iters, ptype=problem,
#                            base=False, savedir=save_dir)

    run_time = datetime.now() - start_time
    print('\n{0} executions completed in: {1}'.format(execs, run_time))
