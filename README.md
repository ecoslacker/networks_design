# NETWORKS DESIGN
Optimal Design of Water Distribution Networks

A framework for optimal design of water distribution systems using Genetic Algorithms (GA) and
Ant Colony Optimization (ACO).
Design includes both layout and pipe sizing of the entire network.

# Requirements

This framework requires:

* Python 2.7
* Numpy
* Matplotlib
* Pyevolve

It is recommended to install **Anaconda** (<https://www.anaconda.com/download/>) or **Python(x,y)** in order to get the requirements.
*Pyevolve* can be installed via *pip* (see below).

# Installation

This framework does not require installation, just copy the directory and run the main script
from a Python interpreter.

## Download

To download this repository you have two options:

1. If you use *git* or another version control software use the command:

```bash
git clone https://github.com/ecoslacker/master-networks.git
```

or

2. Download directly by clicking on the button **Clone or download** above, then click on **Download ZIP**. Extract and run.


## Install a Python environment

You need to install the requirements independently or you can install complete Python distribution such as **Anaconda** (<https://www.anaconda.com/download/>). Be sure to install the Python 2.7 version.

On Windows you can also try with another distribution such as **Python(x,y)**. However, you should consider that this framework is only tested under **Anaconda** and you may need some extra work.

Once the environment is installed, it is recommended to use the *Spyder* IDE to edit and run the Python scripts.

## Install requirements

If you installed **Anaconda**, then the only requirement left is *Pyevolve*. If you want to install this package manually you can install it via *pip* with the following command. 

```bash
pip install pyevolve
```

On Unix-like systems (macOS, Linux) open a *Terminal* and type the command, note that you may need root permissions. On Windows open a *System prompt* and type the command, also you may need to have Administrator privileges.

**IMPORTANT NOTE**: If your system does not have *Pyevolve*, the program will try to install this requirement automagically at the first execution. Your computer should be to be connected to the Internet.

# Usage

Once that you have the program and the framework installed, you can open the main modules.

Open the *optimizer_ga.py* script if you want to optimize the layout or pipe sizes using Genetic Algorithms, or if you want to use the ACO metaheuristic open the *optimizer_aco.py* script.

At the bottom of each file, there is a section starting with the line:

```
if __name__ == "__main__":
```

This line is a special Python condition to indicate the beginning of the main loop of a script. The test code for the functions is placed after this line.

There is no need to edit the code above this line (unless you know exactly what are you doing). Below this line there are some examples you can try.

The *optimizer_ga.py* has two function examples, those can be identified by:

* *optim_dimensions_ga*: to optimize the dimensions of a network using Genetic Algorithms.
* *optim_layout_ga*: to optimize the layout of a network using Genetic Algorithms.

On the other hand the *optimizer_aco.py* has the function:

* *optim_layout_aco*: to optimize the layout of a network using Ant Colony Optimization.

Each function needs different arguments, you can explore each function's documentation to a more detailed description of their possible values.

## Configure

Uncomment the lines of code with the functions you want to use, this is done by removing the number sign (#) at the start of each line of code.

**NOTE**: Remember that in Python language indentation is important, so you need to preserve the number of spaces in multiples of 4 at the start of each line.

If you want to use the layout functions it is recommended to comment the pipe sizing functions and vice versa. This is done to avoid problems at runtime.

Then you need to provide the metaheuristic parameters values to each function, the most important value is the path of the data file for the desired problem instance. The format of the input file required for each problem is described in the next section. 

## Data files

Both metaheuristics need an instance of the problem to be solved, this information is handled by input files.

Files required for, layout optimization problem:

* CSV text file with cartesian coordinates of the problem instance, formatted as indicated by [TSPLIB](<https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/>). Here is an example:

```
NAME : network09
COMMENT : Geem et al. (2000)
TYPE : Network
DIMENSION : 9
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION  
1 0 0
2 100 0
3 0 100
4 200 0
5 100 100
6 0 200
7 200 100
8 100 200
9 200 200
EOF  
```

For the pipe sizing problem:

* An EPANET's **inp** formatted text file with the network configuration and data. Be sure to provide the right units (SI or English Units) to the network. Please consult the [EPANET 2 User Manual](<https://nepis.epa.gov/Adobe/PDF/P1007WWU.pdf>) if necessary. You can create this files with the EPANET 2 software (<https://www.epa.gov/water-research/epanet>).

You can find some examples of these files in the **data** directory.

## Run

Using *Spyder* IDE you just need to execute the script by pressing the *F5* key or by clicking the *Run* button. 

# Examples

In this sections a few examples of how to use the framework are presented.

## Optimize the layout of a network using ACO

If you want to use the ACO metaheuristic you need to open the *optimizer_aco.py* script. Then uncomment the lines for the layout test code.

```python
TO BE INCLUDED
``` 

## Optimize the pipe diameters using GA

If you want to use the GA metaheuristic you need to open the *optimizer_ga.py* script. Then uncomment the lines for the dimensioning test code.

```python
if __name__ == "__main__":

    start_time = datetime.now()

    execs = 20

    # ******** OPTIMIZATION OF NETWORK DIMENSIONS (DIAMETERS) ********
    path = 'data/'
    inp_file = 'TwoLoop.inp'
    directory = 'results_ga/dimen_twoloop/'

    best, pop = optim_dimensions_ga(path, inp_file, execs, pop=200, gen=500,
                                    xover=0.85, mut=0.05, save_dir=directory)

    # Print best
    print('**** OVERALL BEST ****')
    print(best)

    run_time = datetime.now() - start_time
    print('\n{0} executions completed in: {1}'.format(execs, run_time))
```

All the lines that have a number sign (#) at the beginning are commentaries.

The first line is the main loop condition.

The second line saves the start time of the execution and the last two lines are used to print the total run time.

The third line indicates the number of executions (`execs`) that will be performed, in this case is 20.

The next three lines indicate the values if the function arguments, the folder or directory (`path`) where the input file is located, the name of the input file (`inp_file`) and the directory where the results are going to be saved in (`directory`). These directories should exist in advance.

The next two lines are the function to be used, in this case `optim_dimensions_ga` and its arguments. Note that the second line starts aligned to the opening parenthesis `(`.

Finally, the next lines are just for showing (using the `print` function) the best solution obtained from the executions.

# License

Networks Design

Optimal Design of Water Distribution Networks

Copyright (C) 2017 Eduardo Jimenez

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.

