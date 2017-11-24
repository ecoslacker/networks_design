# -*- coding: utf-8 -*-
"""
Read, create and modify Epanet's input text files (*.INP)
Created on Tue Nov 29 01:10:55 2016

@author: ecoslacker
"""
import os
from math import sqrt

ENDLINE = '\n'
# Column positions:
NAMES = 0
VALUES = 1
PIPE_ID = 0
PIPE_INI_NODE = 1
PIPE_END_NODE = 2
PIPE_LENGTH = 3
COORD_X = 1
COORD_Y = 2

# Constants text are used in some places
GLOBAL_EFFICIENCY = ' Global Efficiency  '
GLOBAL_PRICE = ' Global Price       '
DEMAND_CHARGE = ' Demand Charge      '
DURATION = ' Duration           '
HYDRAULIC_TIMESTEP = ' Hydraulic Timestep '
QUALITY_TIMESTEP = ' Quality Timestep   '
PATTERN_TIMESTEP = ' Pattern Timestep   '
PATTERN_START = ' Pattern Start      '
REPORT_TIMESTEP = ' Report Timestep    '
REPORT_START = ' Report Start       '
START_CLOCKTIME = ' Start ClockTime    '
STATISTIC = ' Statistic          '
STATUS = ' Status             '
SUMMARY = ' Summary            '
PAGE = ' Page               '
NODES = ' Nodes              '
LINKS = ' Links              '
UNITS = ' Units              '
HEADLOSS = ' Headloss           '
SPECIFICGRAVITY = ' Specific Gravity   '
VISCOSITY = ' Viscosity          '
TRIALS = ' Trials             '
ACCURACY = ' Accuracy           '
CHECKFREQ = ' CHECKFREQ          '
MAXCHECK = ' MAXCHECK           '
DAMPLIMIT = ' DAMPLIMIT          '
UNBALANCED = ' Unbalanced         '
PATTERN = ' Pattern            '
DEMANDMULTIPLIER = ' Demand Multiplier  '
EMITTEREXPONENT = ' Emitter Exponent   '
QUALITY = ' Quality            '
DIFFUSIVITY = ' Diffusivity        '
TOLERANCE = ' Tolerance          '


class EpanetFile:

    """ EPANET's input file creation class

    :param str filepath, the path to save the file (writting permission req.)
    :paran str filename, the INP Epanet's file name
    """

    def __init__(self, filepath, filename):
        """ Initialize the class
        """
        assert type(filepath) is str, "The path should be string type"
        assert type(filename) is str, "The file name should be string type"
        assert len(filepath) > 0, "Path is empty"
        assert len(filename) > 0, "File name is empty"

        # Initialize arguments
        self.filepath = filepath
        self.filename = filename

        self.title = ''
        self.junctions = []
        self.reservoirs = []
        self.tanks = []
        self.pipes = []
        self.pumps = []
        self.valves = []
        self.demands = []
        self.status = []
        self.patterns = []
        self.curves = []
        self.emitters = []
        self.quality = []
        self.sources = []
        self.reactions = []
        self.mixing = []
        self.coordinates = []
        self.vertices = []
        self.labels = []
        # TODO: Define the next variables with their correct types
        self.tags = ''
        self.controls = ''
        self.rules = ''
        self.energy = [[GLOBAL_EFFICIENCY,
                        GLOBAL_PRICE,
                        DEMAND_CHARGE],
                       [75, 0, 0]]
        self.times = [[DURATION,
                       HYDRAULIC_TIMESTEP,
                       QUALITY_TIMESTEP,
                       PATTERN_TIMESTEP,
                       PATTERN_START,
                       REPORT_TIMESTEP,
                       REPORT_START,
                       START_CLOCKTIME,
                       STATISTIC],
                      ['0:00',
                       '1:00',
                       '0:05',
                       '1:00',
                       '0:00',
                       '1:00',
                       '0:00',
                       '12 am',
                       'NONE']]
        self.report = [[STATUS,
                        SUMMARY,
                        PAGE,
                        NODES,
                        LINKS],
                       ['No', 'Yes', '0', 'All', 'All']]
        self.options = [[UNITS,
                         HEADLOSS,
                         SPECIFICGRAVITY,
                         VISCOSITY,
                         TRIALS,
                         ACCURACY,
                         CHECKFREQ,
                         MAXCHECK,
                         DAMPLIMIT,
                         UNBALANCED,
                         PATTERN,
                         DEMANDMULTIPLIER,
                         EMITTEREXPONENT,
                         QUALITY,
                         DIFFUSIVITY,
                         TOLERANCE],
                        ['CMH', 'H-W', 1, 0.0014, 40, 0.001, 2, 10, 0,
                         'Continue 10', 1, 1.0, 0.5, 'None mg/L', 1, 0.01]]
        self.backdrop = ''

        # EPANET's input file sections
        self.sections1 = ['[TITLE]',
                          '[JUNCTIONS]',
                          '[RESERVOIRS]',
                          '[TANKS]',
                          '[PIPES]',
                          '[PUMPS]',
                          '[VALVES]',
                          '[TAGS]',
                          '[DEMANDS]',
                          '[STATUS]',
                          '[PATTERNS]',
                          '[CURVES]',
                          '[CONTROLS]',
                          '[RULES]',
                          '[ENERGY]',
                          '[EMITTERS]',
                          '[QUALITY]',
                          '[SOURCES]',
                          '[REACTIONS]',
                          '[MIXING]',
                          '[TIMES]',
                          '[REPORT]',
                          '[OPTIONS]',
                          '[COORDINATES]',
                          '[VERTICES]',
                          '[LABELS]',
                          '[BACKDROP]',
                          '[END]']
        self.sections = ['[TITLE]',
                         '[JUNCTIONS]',
                         '[RESERVOIRS]',
                         '[TANKS]',
                         '[PIPES]',
                         '[PUMPS]',
                         '[VALVES]',
                         '[EMITTERS]',
                         '[CURVES]',
                         '[PATTERNS]',
                         '[ENERGY]',
                         '[STATUS]',
                         '[CONTROLS]',
                         '[RULES]',
                         '[DEMANDS]',
                         '[QUALITY]',
                         '[REACTIONS]',
                         '[SOURCES]',
                         '[MIXING]',
                         '[OPTIONS]',
                         '[TIMES]',
                         '[REPORT]']
        # EPANET's section headers for tab delimited text
        self.element = {'Junctions':   [';ID', 'Elev', 'Demand', 'Pattern'],
                        'Reservoirs':  [';ID', 'Head', 'Pattern'],
                        'Tanks':       [';ID', 'Elevation', 'InitLevel',
                                        'MinLevel', 'MaxLevel', 'Diameter',
                                        'MinVol', 'VolCurve'],
                        'Pipes':       [';ID', 'Node1', 'Node2', 'Length',
                                        'Diameter', 'Roughness', 'MinorLoss',
                                        'Status'],
                        'Pumps':       [';ID', 'Node1', 'Node2', 'Parameters'],
                        'Valves':      [';ID', 'Node1', 'Node2', 'Diameter',
                                        'Type', 'Setting', 'MinorLoss'],
                        'Demands':     [';Junction', 'Demand', 'Pattern',
                                        'Category'],
                        'Status':      [';ID', 'Status/Setting'],
                        'Patterns':    [';ID', 'Multipliers'],
                        'Curves':      [';ID', 'X-Value', 'Y-Value'],
                        'Emitters':    [';Junction', 'Coefficient'],
                        'Quality':     [';Node', 'InitQual'],
                        'Sources':     [';Node', 'Type', 'Quality', 'Pattern'],
                        'Reactions':   [';Type', 'Pipe/Tank', 'Coefficient'],
                        'Mixing':      [';Tank', 'Model'],
                        'Coordinates': [';Node', 'X-Coord', 'Y-Coord'],
                        'Vertices':    [';Link', 'X-Coord', 'Y-Coord'],
                        'Labels':      [';X-Coord', 'Y-Coord',
                                        'Label & Anchor Node']}

        # Capitalize each section header and create a list
        self.lsec = [self.sec2cap(x) for x in self.sections1]
        self.lsec.remove('End')
        print(self.lsec)

    def sec2cap(self, word):
        """ Section to capitalize

        Takes a word and remove the first and last characters, then return it
        capitalized, like this: [TITLE] ---> Title

        :param word, an uppercase word inside square brackets
        :return new_word, a capital case word without square brackets
        """

        new_word = word[1:-1]
        return new_word.capitalize()

    def fill_string(self, s, length=20):
        """ Fill string

        Adds blank spaces to the string to get the specified length.

        :param s, original string
        :return newstring, the string with  blank spaces
        """
        assert type(s) is str, "Invalid input type, should be str"
        assert len(s) <= length, "String is too long."

        # If the string already has the length, just return it
        if len(s) is length:
            return s

        l = len(' ' + s)
        newstring = ' ' + s + ' ' * (length - l)
        return newstring

    # --- Some setters ---

    def set_title(self, title):
        """ Assign the title """
        self.title = title
        return True

    def set_junctions(self, junctions):
        """ Set junctions

        Set a list which contains other lists of junctions data

        :param junctions, lists of data for the junctions
        """

        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the junctions
        l = len(self.element['Junctions'])
        assert len(junctions) is l, 'Not enough junction info'
        self.junctions = junctions
        return True

    def set_reservoirs(self, reservoirs):
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the reservoirs
        l = len(self.element['Reservoirs'])
        assert len(reservoirs) is l, 'Not enough reservoirs info'
        self.reservoirs = reservoirs
        return True

    def set_tanks(self, tanks):
        # TODO: Test this fucntion!
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the tanks
        l = len(self.element['Tanks'])
        assert len(tanks) is l, 'Not enough tanks info'
        self.tanks = tanks
        return True

    def set_pipes(self, pipes):
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the pipes
        l = len(self.element['Pipes'])
        assert len(pipes) is l, 'Not enough pipes info'
        self.pipes = pipes
        return True

    def set_pumps(self, pumps):
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the pumps
        l = len(self.element['Pumps'])
        assert len(pumps) is l, 'Not enough pumps info'
        self.pumps = pumps
        return True

    def set_valves(self, valves):
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the valves
        l = len(self.element['Valves'])
        assert len(valves) is l, 'Not enough valves info'
        self.valves = valves
        return True

    def set_coordinates(self, coordinates):
        # TODO: check that all the lists have the same number of elements
        # The number of lists should match the headers for the coordinates
        l = len(self.element['Coordinates'])
        assert len(coordinates) is l, 'Not enough coordinates info'
        self.coordinates = coordinates
        return True

    def modify_list(self, mlist, parameter, value):
        """ Modify a list with the parameter and its value

        A set of two lists acting like a dictionary, the parameter name is set
        to the first list and the value to the second. If the parameter does
        not exist then is created.

        :param mlist, the list to modify
        :param parameter, the name of the parameter to set
        :param value, the value of the parameter
        :return True, if setup success
        """
        assert len(mlist) is 2, "The list should have two elements"
        assert len(mlist[0]) is len(mlist[1]), "List's elements don't match"
        assert type(parameter) is str, "Parameter name should be str"
        assert len(parameter) is not 0, "Invalid input (empty string)"

        # Fill the parameter with blank spaces to get a length of 20
        parameter = self.fill_string(parameter)

        # Insert the value in the index of the parameter, if the parameter
        # is not in the list append both parameter and its value
        if parameter in mlist[NAMES]:
            index = mlist[NAMES].index(parameter)
            mlist[VALUES][index] = value
        else:
            mlist[NAMES].append(parameter)
            mlist[VALUES].append(value)
        return True

    def set_energy(self, parameter, value):
        """ Set energy

        Set data to the energy list.

        :param parameter, the name of the parameter to set
        :param value, the value of the parameter
        :return True, if setup success
        """
        self.modify_list(self.energy, parameter, value)
        return True

    def set_global_efficiency(self, value):
        self.set_energy(GLOBAL_EFFICIENCY, value)
        return True

    def set_global_price(self, value):
        self.set_energy(GLOBAL_PRICE, value)
        return True

    def set_demand_charge(self, value):
        self.set_energy(DEMAND_CHARGE, value)
        return True

    def set_times(self, parameter, value):
        """ Set times

        Set data to the times list.

        :param parameter, the name of the parameter to set
        :param value, the value of the parameter
        :return True, if setup success
        """
        self.modify_list(self.times, parameter, value)
        return True

    def set_report(self, parameter, value):
        """ Set options

        Set data to the report list.

        :param parameter, the name of the parameter to set
        :param value, the value of the parameter
        :return True, if setup success
        """
        self.modify_list(self.report, parameter, value)
        return True

    def set_status(self, value):
        assert type(value) is str, "The report status should be 'str'"
        self.set_report(STATUS, value)
        return True

    def set_summary(self, value):
        assert type(value) is str, "The report summary should be 'str'"
        self.set_report(SUMMARY, value)
        return True

    def set_page(self, value):
        assert type(value) is str, "The report page should be 'str'"
        self.set_report(PAGE, value)
        return True

    def set_report_nodes(self, value):
        assert type(value) is str, "The report nodes should be 'str'"
        self.set_report(NODES, value)
        return True

    def set_report_links(self, value):
        assert type(value) is str, "The report links should be 'str'"
        self.set_report(LINKS, value)
        return True

    def set_options(self, parameter, value):
        """ Set options

        Set data to the options list.

        :param parameter, the name of the parameter to set
        :param value, the value of the parameter
        :return True, if setup success
        """
        self.modify_list(self.options, parameter, value)
        return True

    def set_units(self, value):
        """ Set the project units

        Find the value in the allowed units list and set it up to the project.

        :param value, the value of the units
        :return True, if setup success
        """
        assert (type(value) is str) or (type(value) is int), \
            "Value should be 'str' or 'int'"

        # Available units
        units = ['CFS', 'GPM', 'MGD', 'IMGD', 'AFD',
                 'LPS', 'LPM', 'MLD', 'CMH', 'CMD']

        # Find and set the units
        if type(value) is str:
            value = value.upper()
            if value in units:
                index = units.index(value)
                self.set_options(UNITS, units[index])
            else:
                print("Warning: Invalid units, setting default!")
                return False
        elif type(value) is int:
            if value < len(units):
                self.set_options(UNITS, units[value])
            else:
                print("Warning: Invalid units, setting default!")
                return False
        return True

    def set_headloss_formula(self, value):
        """ Set the project headloss formula

        Find the value in the list of allowed headloss formulas and set it up
        to the project.

        :param value, the value of the headloss formula
        :return True, if setup success
        """
        assert (type(value) is str) or (type(value) is int), \
            "Value should be 'str' or 'int'"

        # Available headloss formulas
        formulas = ['H-W', 'D-W', 'C-M']

        # Find and set the headloss formula
        if type(value) is str:
            value = value.upper()
            if value in formulas:
                index = formulas.index(value)
                self.set_options(HEADLOSS, formulas[index])
            else:
                print("Warning: Invalid headloss formula, setting default!")
                return False
        elif type(value) is int:
            if value < len(formulas):
                self.set_options(HEADLOSS, formulas[value])
            else:
                print("Warning: Invalid headloss formula, setting default!")
                return False
        return True

    def set_specific_gravity(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The specific gravity should be 'int' or 'float'"
        self.set_options(SPECIFICGRAVITY, value)
        return True

    def set_viscosity(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The type of viscosity should be 'int' or 'float'"
        self.set_options(VISCOSITY, value)
        return True

    # TODO: This fucntions should be tested!!!

    def set_trials(self, value):
        assert type(value) is int, "The type of trials should be 'int'"
        self.set_options(TRIALS, value)
        return True

    def set_accuracy(self, value):
        assert type(value) is float, "The type of accuracy should be 'float'"
        self.set_options(ACCURACY, value)
        return True

    def set_checkfreq(self, value):
        assert type(value) is int, "The type of CHECKFREQ should be 'int'"
        self.set_options(CHECKFREQ, value)
        return True

    def set_maxcheck(self, value):
        assert type(value) is int, "The type of MAXCHECK should be 'int'"
        self.set_options(MAXCHECK, value)
        return True

    def set_damplimit(self, value):
        assert type(value) is int, "The type of DAMPLIMIT should be 'int'"
        self.set_options(DAMPLIMIT, value)
        return True

    def set_unbalanced(self, value):
        assert type(value) is str, "The type of unbalanced should be 'str'"
        self.set_options(UNBALANCED, value)
        return True

    def set_pattern(self, value):
        assert type(value) is int, "The type of pattern should be 'int'"
        self.set_options(PATTERN, value)
        return True

    def set_demand_multiplier(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The type of demand multiplier should be 'int' or 'float'"
        self.set_options(DEMANDMULTIPLIER, value)
        return True

    def set_emitter_exponent(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The type of emitter exponent should be 'int' or 'float'"
        self.set_options(EMITTEREXPONENT, value)
        return True

    def set_quality(self, value):
        assert type(value) is str, "The type of quality should be 'str'"
        self.set_options(QUALITY, value)
        return True

    def set_diffusivity(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The type of diffusivity should be 'int' or 'float'"
        self.set_options(DIFFUSIVITY, value)
        return True

    def set_tolerance(self, value):
        assert (type(value) is float) or (type(value) is int), \
            "The type of tolerance should be 'int' or 'float'"
        self.set_options(TOLERANCE, value)
        return True

    # TODO: CREATE THE SETTERS FOR THE OTHER VARIABLES!

    def create_content(self):
        """ Create content

        Creates the content of the text file using all the data provided with
        the setter methods and the class properties.
        """

        self.data = {'Junctions':   self.junctions,
                     'Reservoirs':  self.reservoirs,
                     'Tanks':       self.tanks,
                     'Pipes':       self.pipes,
                     'Pumps':       self.pumps,
                     'Valves':      self.valves,
                     'Demands':     self.demands,
                     'Status':      self.status,
                     'Patterns':    self.patterns,
                     'Curves':      self.curves,
                     'Emitters':    self.emitters,
                     'Quality':     self.quality,
                     'Sources':     self.sources,
                     'Reactions':   self.reactions,
                     'Mixing':      self.mixing,
                     'Coordinates': self.coordinates,
                     'Vertices':    self.vertices,
                     'Labels':      self.labels,
                     'Title':       self.title,
                     'Tags':        self.tags,
                     'Controls':    self.controls,
                     'Rules':       self.rules,
                     'Energy':      self.energy,
                     'Times':       self.times,
                     'Report':      self.report,
                     'Options':     self.options,
                     'Backdrop':    self.backdrop}
        self.content = ''
        keys = self.element.keys()
        i = 0

        # Process each element of the list of sections and the actual data.
        # IMPORTANT: This block of code may be confusing, the only thing it
        # does is to put together the headers and the actual numeric values in
        # a single string. This works with the 'data' dictionary created
        # before.
        for item in self.lsec:
            if item in keys:
                # print(item + ' is in elements!')
                headers = '\t'.join(self.element[item]) + ENDLINE
                # print(self.element[item])

                # This adds the section title and the comment headers for some
                # tab delimited data, as junctions and pipes.
                text = self.sections1[i] + ENDLINE + headers

                # Get the proper array of data
                d = self.data[item]

                if len(d) is 0:
                    self.content = self.content + text + ENDLINE
                    i += 1
                    continue

                # WARNING! This assumes that 'Junctions' and 'Reservoirs' are
                # always in the elements with delimited text
                if item == 'Junctions' or item == 'Reservoirs':
                    # Some data like 'Junctions' and 'Reservoirs' could have
                    # a pattern
                    pattern_provided = False

                    # The pattern is always the last array (index -1),
                    # check if it's not empty and if it has the same
                    # length as the first array.
                    if (len(d[0]) == len(d[-1])) and (len(d[-1]) != 0):
                        pattern_provided = True

                    # Iterate each row (or element of the arrays)
                    for row in range(len(d[0])):
                        l = []

                        # TODO: WARNING! This assumes that all the arrays
                        # have the same length. Check first !!!

                        # Iterate each column (or array) except the last one
                        for col in range(len(d) - 1):
                            l.append(str(d[col][row]))
                        if pattern_provided:
                            l.append(d[-1][row])
                        else:
                            l.append(' ')
                        line = '\t'.join(l) + '\t;' + ENDLINE
                        text = text + line
                else:
                    for row in range(len(d[0])):
                        l = []

                        # TODO: WARNING! This assumes that all the arrays
                        # have the same length. Check first !!!

                        # Iterate each column (or array)
                        for col in range(len(d)):
                            l.append(str(d[col][row]))
                        line = '\t'.join(l) + '\t;' + ENDLINE
                        text = text + line
            else:
                # print(item + ' not in list of elements')
                text = self.sections1[i] + ENDLINE
                d = self.data[item]
                if type(d) is list:
                    # TODO: Define this block
                    for row in range(len(d[0])):
                        l = []
                        for col in range(len(d)):
                            l.append(str(d[col][row]))
                        line = '\t'.join(l) + ENDLINE
                        text = text + line
#                    self.content = self.content + text + ENDLINE
#                    i += 1
#                    continue
                elif type(d) is str:
                    text = text + self.data[item] + ENDLINE
            i += 1
            self.content = self.content + text + ENDLINE
        self.content = self.content + self.sections1[-1]
        print(self.content)
        return self.content

    def calc_length(self):
        """ Calculate pipes length

        Uses the nodes coordinates to calculate the pipes length, this function
        requires the pipes and coordinates to be setted first.
        """
        for i in range(len(self.pipes[PIPE_ID])):
            # Get initial and final node
            # Substract 1 unit because the Python start count at 0 and
            # Epanet nodes start at 1.
            inode = self.pipes[PIPE_INI_NODE][i] - 1
            fnode = self.pipes[PIPE_END_NODE][i] - 1
            print("Link {0} starts at node {1} ends at {2}".format(i, inode,
                  fnode))

            # Get the coordinates
            x1 = self.coordinates[COORD_X][inode]
            y1 = self.coordinates[COORD_Y][inode]
            x2 = self.coordinates[COORD_X][fnode]
            y2 = self.coordinates[COORD_Y][fnode]

            # Calculate the euclidean 2D distance between the points
            dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Update the length
            self.pipes[PIPE_LENGTH][i] = dist

    def write_file(self):
        """ Write file

        Creates a text file formatted as an EPANET's input (INP) using the
        information provided by user.
        """
        name = os.path.join(self.filepath, self.filename)
        with open(name, 'w') as f:
            f.write(self.create_content())
        return True


# *** END OF THE CLASSES ***

if __name__ == "__main__":

    # *** THIS IS SOME TEST CODE ***

    path = 'data/'

    # **** The 3x3 network ****
    name = 'Network09_new.inp'
    title = '3x3 network Geem et al. (2000)'
    # Junctions
    j = [[1, 2, 3, 4, 5, 6, 7, 8],  # ID
         [0, 0, 0, 0, 0, 0, 0, 0],  # Elevation
         [10, 20, 10, 20, 10, 20, 10, 20],  # Demand (L/s)
         []]  # Pattern
    # Reserviors
    r = [[9], [50], []]  # ID, Head, Pattern
    # Pipes
    p = [[1, 2, 3, 4, 5, 6, 7, 8],  # ID
         [1, 4, 7, 2, 5, 6, 6, 8],  # Initial node
         [2, 2, 4, 5, 8, 8, 3, 9],  # End node
         [100] * 12,    # Length
         [25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4,
          25.4],        # Diameter
         [130] * 12,    # Roughness coefficient
         [0] * 12,      # Minor loss
         ['Open'] * 12]
    # Coordinates
    c = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 100, 0, 200, 100, 0, 200, 100, 200],
         [0, 0, 100, 0, 100, 200, 100, 200, 200]]

#    # *** The Two Loop network ***
#    name = 'TwoLoop_new.inp'
#    title = 'Two Loop network Alperovits & Shamir (1977)'
#    # Junctions
#    j = [[2, 3, 4, 5, 6, 7],  # ID
#         [150, 160, 155, 150, 165, 160],  # Elevation
#         [100, 100, 120, 270, 330, 200],  # Demand
#         []]  # Pattern
#    # Reservoirs
#    r = [[1], [210], []]  # ID, Head, Pattern
#    # Pipes
#    p = [[1, 2, 3, 4, 5, 6, 7, 8],  # ID
#         [1, 2, 2, 4, 4, 6, 3, 5],  # Initial node
#         [2, 3, 4, 5, 6, 7, 5, 7],  # End node
#         [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],  # Length
#         [457, 254, 406, 101, 406, 254, 254, 25.4],  # Diameter
#         [130, 130, 130, 130, 130, 130, 130, 130],   # Roughness coefficient
#         [0, 0, 0, 0, 0, 0, 0, 0],  # Minor loss
#         ['Open', 'Open', 'Open', 'Open', 'Open', 'Open', 'Open', 'Open']]
#    # Coordinates
#    c = [[2, 3, 4, 5, 6, 7, 1],
#         [3514.19, 1314.79, 3514.19, 1314.79, 3525.65, 1326.25, 5396.44],
#         [7722.92, 7722.92, 5809.90, 5821.36, 3770.88, 3770.88, 7718.45]]

    # Class instance
    inp = EpanetFile(path, name)
    inp.set_title(title)
    inp.set_junctions(j)
    inp.set_reservoirs(r)
    inp.set_pipes(p)
    inp.set_coordinates(c)
    inp.set_units(8)  # CMH (Cubic meters per hour)
    inp.set_headloss_formula('H-W')
    inp.write_file()
