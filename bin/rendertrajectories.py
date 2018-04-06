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
# Renders trajectories in an input-file (console program)
#
# This program opens a FFE.png file and renders the trajectories in this file to a plot. This program
# uses minimal logging.
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


# Function: Prints help page
# --------------------------------------------------------------------------------------------------------------------
def printHelpPage():
    print("USAGE: script.py [options] --input-file <file>")
    print("")

    print("Order of options is not important. The input file is mandatory.")
    print("")

    print("Switches:")
    print("\t--help:\t\t\t\tShows this help page.")
    print("\t--debug:\t\t\tDebug mode.")
    print("\t--silent:\t\t\tSilent mode.")
    print("\t--output-file <file>\t\tSave output to a file rather than screen.")
    print("\t--output-style <style>\t\tUse render style <style>. Possible values: linear, spline, skeleton. "
          "Default: linear.")
    print("\t--useinlet #\t\t\tUse inlet # (zero based!) as origin point (if present). Default: Uses inlet nearest "
          "to left/middle point.")
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

# How to render trajectories?
outputstyle = "linear"

# Use inlet
useinlet = -1               # Uses only inlet # for evaluating paths; default (-1): use every inlet.

# No parameters given?
if len(sys.argv) == 1:
    printHelpPage()
    sys.exit(1)

# Try to find all the arguments
try:
    # All the options to recognize
    opts, args = getopt.getopt(sys.argv[1:], "", ["help", "debug", "silent", "input-file=",
                                                  "output-file=", "output-style=", "useinlet="])

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
    elif opt == "--output-style":
        if str(arg) in ["linear", "spline", "skeleton"]:
            outputstyle = arg
        else:
            print("Outputstyle '%s' not recognized." % str(arg))
            printHelpPage()
            sys.exit(3)
    elif opt == "--useinlet":
        if 0 <= int(arg):
            useinlet = int(arg)
        else:
            print("Useinlet should be a positive integer. '%s' was given." % str(arg))


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
logger.info(u"####### Render trajectories #######")
logger.info(u"Render trajectories on '%s'", inputfile)

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


# Step 3: Rendering the trajectories
# --------------------------------------------------------------------------------------------------------------------
# Let the origin be the first inlet if exists (origin is in real dimensions, i.e. mm!)
origin = (0, 0)

if "Inlets" in ffedata:
    # Inlets
    inlets = np.array([(item[0], item[1]) for item in ffedata["Inlets"]])

    # Useinlet given and in the range of given inlets? Well, then only use this one start point
    if 0 <= useinlet < len(inlets):
        logger.info(u"Use inlet %d for origin point.", useinlet)
        origin = (inlets[useinlet][0], inlets[useinlet][1])

    # or just select the point nearest to center of the left side
    else:
        # First sort the points by distance from the center of the left side (0, zoneheight/2)
        inlets = ffe.sortCoordinatesByDistanceToPoint(inlets, (0, int(ffedata["Physical zone dimensions"][1]/2)))

        # Calculate new origin from nearest point (first point in array)
        origin = (inlets[0][0], inlets[0][1])


# These are the "Tableau 10" colors as RGB. Changed the order a little bit.
# Source: http://tableaufriction.blogspot.ca/2012/11/finally-you-can-use-tableau-data-colors.html
tableau10 = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14),
             (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)

# Get axes
ax = plt.axes()

# Set axes ticks
ax.set_xticks(np.arange(-50, 50, 5))
ax.set_xticks(np.arange(-50, 50, 1), minor=True)
ax.set_yticks(np.arange(-50, 50, 5))
ax.set_yticks(np.arange(-50, 50, 1), minor=True)

# Set axes limits
ax.set_xlim([0-origin[0], float(ffedata["Physical zone dimensions"][0])-origin[0]])
ax.set_ylim([0-origin[1], float(ffedata["Physical zone dimensions"][1])-origin[1]])

# Set ticks
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Set aspect ratio to 1:1
ax.set_aspect('equal')

# Invert y-axis
ax.invert_yaxis()

# Add grid
plt.grid()

# Draw axis lines
plt.axhline(y=0, color='black', ls='dashed', linewidth=3, zorder=0)
plt.axvline(x=0, color='black', ls='dashed', linewidth=3, zorder=1)

# Handles for legend
handles = []

# Process each trajectory
for index, trajectory in enumerate(trajectories):
    # Get the skin of the trajectory
    skin = ffe.getSkinOfTrajectory(trajectory)

    # Add handle
    handles.append(matplotlib.lines.Line2D([0, 1], [0, 1], color=tableau10[index % len(tableau10)],
                                           linewidth=5, alpha=0.5))

    # How to draw the trajectories... skeleton?
    if outputstyle == 'skeleton':
        logger.info(u"Skeleton rendering of trajectory %d...", index+1)

        # x-values for width
        xvalues = [float(t[0] * resolution[0]) - origin[0] for t in skin]
        # y-values for width
        yvalues = [float(t[1] * resolution[1]) - origin[1] for t in skin]

        # Number of xvalues should be number of yvalues and it should be a multiply of 2 for drawing the lines properly
        if len(xvalues) == len(yvalues) and (len(xvalues) % 2) == 0:
            # Number of values devided by two
            halflen = int(len(xvalues)/2)

            for l in xrange(halflen):
                plt.plot([xvalues[l], xvalues[halflen*2-l-1]], [yvalues[l], yvalues[halflen*2-l-1]],
                         lw=1, color=tableau10[index % len(tableau10)])

        # Plot all width points (small)
        plt.scatter(xvalues, yvalues, s=1, color=tableau10[index % len(tableau10)], zorder=100 + index)

        # x-values for trajectory
        xvalues = [float(t[0] * resolution[0]) - origin[0] for t in trajectory]
        # y-values for trajectory
        yvalues = [float(t[1] * resolution[1]) - origin[1] for t in trajectory]

        # Plot trajectory points itself
        plt.plot(xvalues, yvalues, color=tableau10[index % len(tableau10)], zorder=100 + index)
        plt.scatter(xvalues, yvalues, color=tableau10[index % len(tableau10)], zorder=100 + index)

    # or splines?
    elif outputstyle == 'spline':
        logger.info(u"Spline rendering of trajectory %d...", index+1)

        # Convert to real-world dimensions
        data = np.array([(float(t[0] * resolution[0]) - origin[0],
                          float(t[1] * resolution[1]) - origin[1]) for t in skin])

        # Resolution for interpolation
        newres = np.arange(0, 1.01, 0.01)

        # Interpolate as spline (cubic, k=3)
        spline, _ = scipy.interpolate.splprep(data.transpose(), s=0)

        # Get spline points of spline
        splinepoints = scipy.interpolate.splev(newres, spline)

        # Convert to vertices
        vertices = [zip(splinepoints[0], splinepoints[1])]

        # Patch collection
        pc = matplotlib.collections.PolyCollection(vertices, color=tableau10[index % len(tableau10)],
                                                   zorder=100 + index,
                                                   alpha=0.5)

        # Add the collection
        ax.add_collection(pc)

        # Draw the weighted-y lines
        weightedy = ffe.calculateWeightedYOfTrajectory(trajectory) * resolution[1] - origin[1]
        plt.axhline(y=weightedy, color=tableau10[index % len(tableau10)], ls='dashed', linewidth=3, zorder=50 + index)

    # otherwise just use linear drawing (skin)
    elif outputstyle == 'linear':
        logger.info(u"Linear rendering of trajectory %d...", index+1)

        # x-values
        xvalues = [float(t[0] * resolution[0]) - origin[0] for t in skin]
        # y-values
        yvalues = [float(t[1] * resolution[1]) - origin[1] for t in skin]

        # Convert the skin to vertices
        vertices = [zip(xvalues, yvalues)]

        # Patch collection
        pc = matplotlib.collections.PolyCollection(vertices, color=tableau10[index % len(tableau10)], zorder=100+index,
                                                   alpha=0.5)

        # Add the collection
        ax.add_collection(pc)

        # Draw the weighted-y lines
        weightedy = ffe.calculateWeightedYOfTrajectory(trajectory) * resolution[1] - origin[1]
        plt.axhline(y=weightedy, color=tableau10[index % len(tableau10)], ls='dashed', linewidth=3, zorder=50+index)




# Set title
ax.set_title("Trajectories of %s" % inputfile, fontsize=18)
ax.set_xlabel('x [mm]', fontsize=16, fontweight='bold')
ax.set_ylabel('y [mm]', fontsize=16, fontweight='bold')

# Set legend for trajectories
ax.legend(handles, ["Trajectory %d" % (i+1) for i in xrange(len(trajectories))], loc='lower left').set_zorder(200)

# Show the plot
if len(outputfile) > 0:
    plt.savefig(outputfile)
    logger.info(u"Saved plot to file %s.", outputfile)
else:
    plt.show()
    logger.info(u"Plotted.")

# Final logging
logger.info(u"####### Render trajectories end #######")




