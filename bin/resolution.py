#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2017 by Sven Kochmann
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Description:
# Renders the resolution of an image along the x-axis (console program)
#
# This program opens a FFE.png file, and extracts and renders the resolution in this file to a plot.
# This program uses minimal logging.
#

# Import modules
import cv2                          # OpenCV
import numpy as np                  # Numpy - You always need this.
import ffe                          # frequently-used function script
import sys                          # Sys functions
import logging                      # Logging functions
import getopt                       # Get and parse command-line arguments
import matplotlib.pyplot as plt     # For plotting intermediate graphs (debug)
import matplotlib.collections
import scipy
import scipy.interpolate
import itertools                    # For iteration/combination purposes
from statsmodels.nonparametric.smoothers_lowess import lowess  # For smoothing


# Function: Prints help page
# --------------------------------------------------------------------------------------------------------------------
def printHelpPage():
    print("USAGE: script.py [options] --input-file <file>")
    print("")

    print("Order of options is not important. The input file is mandatory.")
    print("")

    print("Switches:")
    print("\t--help:\t\t\t\t\tShows this help page.")
    print("\t--debug:\t\t\t\tDebug mode.")
    print("\t--silent:\t\t\t\tSilent mode.")
    print("\t--output-file <file>\tSave output to a file rather than screen.")
    print("\t--smooth <value>:\t\tAdds a smoothed line to the plot generated with the fraction of data specified with"
          " this parameter. Range: 0.00-1.00. (default: no smoothing).")
    print("")

# Step 0: Parse command-line arguments
# --------------------------------------------------------------------------------------------------------------------
# Input/Outputfile
inputfile = ""
outputfile = ""

# Debugmode
debugmode = False

# Silent mode
silentmode = False

# Smoothing of resolution lines
smoothing = False
smoothfrac = 0.25

# No parameters given?
if len(sys.argv) == 1:
    printHelpPage()
    sys.exit(1)

# Try to find all the arguments
try:
    # All the options to recognize
    opts, args = getopt.getopt(sys.argv[1:], "", ["help", "debug", "silent", "input-file=",
                                                  "output-file=", "smooth="])

# Found something unexpected? Display help page!
except getopt.GetoptError as parametererror:
    print("ERROR: %s." % parametererror.msg)
    printHelpPage()
    sys.exit(2)

# Otherwise, collect the user input
for opt, arg in opts:
    if opt == '--help':
        printHelpPage()
        sys.exit(0)
    elif opt == "--input-file":
        inputfile = arg
    elif opt == "--output-file":
        outputfile = arg
    elif opt == "--debug":
        debugmode = True
    elif opt == "--silent":
        silentmode = True
    elif opt == "--smooth":
        if 0.00 <= float(arg) <= 1.00:
            smoothfrac = float(arg)
            smoothing = True
        else:
            print("Smooth must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
            sys.exit(0)

# No input file given?
if len(inputfile) == 0:
    print("No input file given.")
    printHelpPage()
    sys.exit(1)


# Step 1: Setup logging
# --------------------------------------------------------------------------------------------------------------------
# Create logger object for this script
logger = logging.getLogger('FFE')

# Set level of information
logger.setLevel(logging.DEBUG)

# Create log file handler which records everything; append the new information
fh = logging.FileHandler('evaluating.log', mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t%(filename)s\t%(levelname)s\t%(lineno)d\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Create console handler which shows INFO and above (WARNING, ERRORS, CRITICALS, ...)
if not silentmode:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Start the program
logger.info(u"####### Extract and render resolutions #######")
logger.info(u"Render resolutions on '%s'", inputfile)

# Debug mode?
if debugmode:
    ffe.enableDebugMode()


# Step 2: Open file and read data
# --------------------------------------------------------------------------------------------------------------------
# Read source image
inputimage = cv2.imread(inputfile, cv2.IMREAD_COLOR)

# Error?
if inputimage is None:
    logger.error(u"Input file could not be read.")
    sys.exit(3)

# Shape of image
imageheight, imagewidth, imagebits = inputimage.shape

# Read data
ffedata = ffe.loadDictionaryFromPng(inputfile)

# Are there trajectories in file's data?
if "Inner separation zone" not in ffedata:
    logger.error(u"No inner separation zone in file.")
    sys.exit(-1)

# Extracts the separation zone from the image
zoneimage, transmatrix = ffe.extractZoneFromImage(inputimage, ffedata["Inner separation zone"], 10)

# Shape of zone image
zoneheight, zonewidth, zonebits = zoneimage.shape

# Are there trajectories in file's data?
if "Trajectories" not in ffedata:
    logger.error(u"No trajectories in file.")
    sys.exit(-1)

# Read trajectories
trajectories = ffedata["Trajectories"]

# Sort trajectories
trajectories = ffe.sortTrajectoriesByWeightedCoordinates(trajectories)

# This is the resolution for both axes in mm per pixel
resolution = (0.15, 0.15)

# Given in input file?
if "Physical zone dimensions" in ffedata:
    resolution = (float(ffedata["Physical zone dimensions"][0]) / float(zonewidth),
                  float(ffedata["Physical zone dimensions"][1]) / float(zoneheight))

# These are the "Tableau 10" colors as RGB. Changed the order a little bit.
# Source: http://tableaufriction.blogspot.ca/2012/11/finally-you-can-use-tableau-data-colors.html
tableau10 = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14),
             (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)


# Step 3: Prepare plot
# --------------------------------------------------------------------------------------------------------------------
# Let the origin be the first inlet if exists (origin is in real dimensions, i.e. mm!)
origin = (0, 0)

if "Inlets" in ffedata:
    # Inlets
    inlets = np.array([(item[0], item[1]) for item in ffedata["Inlets"]])

    # First sort the points by distance from the center of the left side (0, zoneheight/2)
    inlets = ffe.sortCoordinatesByDistanceToPoint(inlets, (0, int(ffedata["Physical zone dimensions"][1]/2)))

    # Calculate new origin from nearest point (first point in array)
    origin = (inlets[0][0], inlets[0][1])

# Get axes
ax = plt.axes()

# Set axes ticks
ax.set_xticks(np.arange(-50, 50, 5))
ax.set_xticks(np.arange(-50, 50, 1), minor=True)
ax.set_yticks(np.arange(-50, 50, 1))
ax.set_yticks(np.arange(-50, 50, 0.1), minor=True)

# Set axes limits
ax.set_xlim([0-origin[0], float(ffedata["Physical zone dimensions"][0])-origin[0]])
ax.set_ylim(-0.1, 10)

# Set ticks
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Add grid
plt.grid()

# Draw axis lines
plt.axvline(x=0, color='black', ls='dashed', linewidth=3, zorder=1)

# Handles for legend
handles = []

# Step 4: Prepare combinations
# --------------------------------------------------------------------------------------------------------------------
# Create all combinations of trajectories (gives a list with (index1, index2) tuples)
combinations = list(itertools.combinations(xrange(len(trajectories)), 2))

# This is the range. The maximum precision is one pixel
evalrange = np.arange(0, zonewidth, 1)

# Calculate a drawing range based on evalrange
drawrange = evalrange * resolution[0] - origin[0]

# Max resolution
maxres = 1.0

# Drawn combos
drawcombos = []

# For each each combination
for combo in combinations:
    # Calculate for each x in drawrange a resolution of both trajectories
    R = [ffe.getResolutionOfTrajectories(trajectories[combo[0]], trajectories[combo[1]], x) for x in evalrange]

    # Change maxres if necessary
    if max(R) > maxres:
        maxres = max(R)

    # Put resolution and x-range in one array and transpose it; use drawrange instead of evalrange
    data = np.array([drawrange, R]).transpose()

    # Remove points with zero resolution
    data = data[data[:, -1] > 0]

    # Datapoints left?
    if len(data) > 0:
        # Transpose again
        data = data.transpose()

        # Smoothing?
        if smoothing:
            smoothed = lowess(data[1], data[0], frac=smoothfrac)
            plt.plot(smoothed[:, 0], smoothed[:, 1], color=tableau10[len(drawcombos) % len(tableau10)], lw=5,
                     zorder=50 + len(drawcombos), linestyle='--')

        # Plot the resolution
        plt.plot(data[0], data[1], lw=2, color=tableau10[len(drawcombos) % len(tableau10)],
                 zorder=100 + len(drawcombos))

        # Add handle for legend
        handles.append(matplotlib.lines.Line2D([0, 1], [0, 1], color=tableau10[len(drawcombos) % len(tableau10)],
                                               linewidth=5, alpha=0.5))

        # Add to combo
        drawcombos.append(combo)


# Reconfigure y-axis
ax.set_yticks(np.arange(-50, 50, 1))
ax.set_yticks(np.arange(-50, 50, maxres/10.0), minor=True)
ax.set_ylim(-0.1, maxres)


# Step 5: Finish plot
# --------------------------------------------------------------------------------------------------------------------
# Set title
ax.set_title("Resolutions of %s" % inputfile, fontsize=18)
ax.set_xlabel('x [mm]', fontsize=16, fontweight='bold')
ax.set_ylabel('Resolution', fontsize=16, fontweight='bold')

# Set legend for trajectories
ax.legend(handles, ["R for %d <-> %d" % (combo[0]+1, combo[1]+1) for combo in drawcombos],
          loc='upper left').set_zorder(200)

# Show the plot
if len(outputfile) > 0:
    plt.savefig(outputfile)
    logger.info(u"Saved plot to file %s.", outputfile)
else:
    plt.show()
    logger.info(u"Plotted.")

# Final logging
logger.info(u"####### Render resolutions end #######")


