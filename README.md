# NETWORKS DESIGN
Optimal Design of Water Distribution Networks

A framework for optimal design of water distribution systems using Genetic Algorithms (GA) and
Ant Colony Optimization (ACO).
Design includes both layout and pipe sizing of the entire network.

## Requirements

This framework requires:

* Python 2.7
* Numpy
* Matplotlib
* Pyevolve

It is recommended to install **Anaconda** or **Python(x,y)** in order to get the requirements.
*Pyevolve* can be installed via *pip* (see below).

## Installation

This framework does not require installation, just copy the directory and run the main script
from a Python interpreter.

### Download

To download this repository you have two options:

1. If you use *git* or another version control software use the command:

```bash
git clone https://github.com/ecoslacker/master-networks.git
```

or

2. Download directly by clicking on the button **Clone or download** above, then click on **Download ZIP**. Extract and run.


### Install a Python environment

You need to install the requirements independently or you can install complete Python distribution such as **Anaconda** (<https://www.anaconda.com/download/>). Be sure to insall the Python 2.7 version.

On Windows you can also try with another distribution such as **Python(x,y)**. However you should consider that this framework is only tested under **Anaconda** and you may need some extra work.  

### Install requirements

If you installed **Anaconda**, then the only requirement left is *Pyevolve*. This one you can install via *pip* with the following command. Also, you can install the *pyevolve* package from the **Anaconda Navigator**. 

On Unix-like systems (macOS, Linux) open a *Terminal* and type the command, note that you may need root permissions. On Windows open a *System prompt* and type the command. 

```bash
pip install pyevolve
```

## Usage

Once that you have the program and the framework installed, you can open the main modules.

Open the *optimizer_ga.py* script if you want to optimize the layout or pipe sizes using Genetic Algorithms, or if you want to use the ACO metaheuristic open the *optimizer_aco.py* script.

### Data files

The format of the data files is special.

### Configure

Uncomment the lines of the metaheuristic to use, this is done by removing the # symbols.

Provide the path of the data file for the desired problem instance.

Configure the metaheuristic paramenters.

### Run

Execute the script by pressing the *F5* key or by clicking the *Run* button. 

## Examples

In this sections a few examples of how to use the framework are presented.

## Optimize the layout of a network using ACO

If you want to use the ACO metaheuristic you need to open the *optimizer_aco.py* script. Then uncomment the lines for the layout test code.

## Optimize the pipe diameters using GA

If you want to use the GA metaheuristic you need to open the *optimizer_ga.py* script. Then uncomment the lines for the dimensioing test code.

## License

Master Networks

Optimal Design of Water Distribution Networks

Copyright (C) 2017 Eduardo Jimenez

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

