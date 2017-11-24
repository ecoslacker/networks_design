# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:31:17 2017

@author: ecoslacker
"""
import epanet.epamodule as em
import os.path
EXT_INPUT = 'inp'
EXT_REPORT = 'rpt'


class Network:

    def __init__(self, basepath, filename):
        """ Initialize the network object

        :param basepath, working directory (should end with '/')
        :param filename, input file name (with .inp extension)
        """
        self.path = basepath
        self.inputfile = filename
        self.outputfile = filename[-4:] + '.' + EXT_REPORT
        self.name = None
        self.pipes = None  # Links
        self.nodes = None
        self.reserviors = None
        self.coordinates = None
        self.x_coordinates = None
        self.y_coordinates = None
        self.pressure = []
        self.velocity = []
        self.lengths = []

    def check_file(self):
        """ Check file

        Checks if the input file exist
        """
        exist = os.path.isfile(self.path + self.inputfile)
        return exist

    def open_network(self):
        """ Open network

        Open the EPANET toolkit & hydraulics solver
        """
        fi = self.path + self.inputfile
        fo = self.path + self.outputfile

        if self.check_file():
            em.ENopen(fi, fo)
            em.ENopenH()
        else:
            print('File: {0} does not exist'.format(fi))

        return True

    def close_network(self):
        """ Close network

        Close hydraulics solver & EPANET toolkit
        """
        em.ENcloseH()
        em.ENclose()

    def initialize(self):
        """ Get info

        Retrieve the information of the network from EPANET
        """
        self.nodes = em.ENgetcount(em.EN_NODECOUNT)
        self.pipes = em.ENgetcount(em.EN_LINKCOUNT)

    def change_diameters(self, diameters):
        """ Change diameters

        Change the original diameters of the network by the specified ones
        """
        # Check that the number of diameters is equal to the number of pipes
        assert len(diameters) is self.pipes, \
            "The number of diameters anf pipes don't match"

        for i in range(self.pipes):
            # print("Setting diameter: {0}".format(diameters[i]))
            em.ENsetlinkvalue(i+1, em.EN_DIAMETER, diameters[i])

    def simulate(self):
        """ Simulate

        Runs a hydraulic simulation of the network with EPANET
        """
        # Initialize the hydraulic solver
        em.ENinitH(em.EN_NOSAVE)
        self.pressure = []
        self.velocity = []
        self.lengths = []

        # Hydraulic simulation
        while True:
            em.ENrunH()

            # Retrieve hydraulic results for time t
            for i in range(self.nodes):
                # Get the pressure
                p = em.ENgetnodevalue(i+1, em.EN_PRESSURE)
                self.pressure.append(p)
            for i in range(self.pipes):
                # Get the velocity of the flow in the pipe
                v = em.ENgetlinkvalue(i+1, em.EN_VELOCITY)
                l = em.ENgetlinkvalue(i+1, em.EN_LENGTH)
                self.velocity.append(v)
                self.lengths.append(l)
            tstep = em.ENnextH()
            if tstep <= 0:
                break

        return self.pressure, self.velocity, self.lengths

    def change_init_nodes(self):
        """ Change initial nodes of each pipe

        Change the original initial nodes of each pipe
        """
        for i in range(1, self.pipes+1):
            s, e = em.ENgetlinknodes(i)
            print("Link: {0} ({1}-{2})".format(i, s, e))
        return

    def save_inp_file(self, filename):
        em.ENsaveinpfile(filename)
        return


# *** END OF THE CLASSES ***

if __name__ == "__main__":

    # *** THIS IS SOME TEST CODE ***
    path = 'data/'
    infile = 'TwoLoop.inp'
    mynet = Network(path, infile)
    mynet.open_network()
    mynet.initialize()
    mynet.simulate()
    mynet.change_init_nodes()
    mynet.close_network()

    print(mynet.pressure)
    print(mynet.velocity)
