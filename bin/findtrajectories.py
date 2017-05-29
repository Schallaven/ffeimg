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

# Import modules
import cv2              # OpenCV
import numpy as np      # Numpy - You always need this.
import ffe              # frequently-used function script
import sys              # Sys functions
import logging          # Logging functions
import getopt           # Get and parse command-line arguments


# Function: Prints help page
# --------------------------------------------------------------------------------------------------------------------
def printHelpPage():
    print("USAGE: script.py [options] --input-file <file>")
    print("")

    print("Order of options is not important. The input file is mandatory.")
    print("")

    print("Switches:")
    print("\t--help:\t\t\t\t\t\tShows this help page.")
    print("\t--debug:\t\t\t\t\tActivate debug mode output of intermediate pictures as debugNNN.png and preview.")
    print("\t--silent:\t\t\t\t\tDo not show anything in the console (except parameter errors).")
    print("")

    print("General options:")
    print("\t--input-file <file>\t\t\tInput file. <file> should be a PNG-file.")
    print("\t--channel <channel>\t\t\tUses only <channel> for feature finding. Default: All channels are used. "
          "<channel> can be blue, green, or red.")
    print("\t--zone-border <width>\t\tWidth of border, which will be blackened. Default: 20")
    print("\t--threshbin <tresh>\t\t\tBias for binary picture. Range: 0-255. Default: 45.")
    print("\t--noendpoints\t\t\tRemove endpoints from trajectories")
    print("\t--useinlet #\t\t\tUses only inlet # (zero based!) as start point (if present). Default: Uses every inlet.")
    print("\t--minpoints #\t\t\tMinimum points a trajectory must have. Default: 0.")
    print("")

    print("Options for comparing trajectories:")
    print("\t--maxoverlap <overlap>\t\tMaximum overlap two trajectories can have before they are considered "
          "identical. Range: 0.00-1.00. Default: 0.75.")
    print("\t--useHausdorff\t\t\t\tUses Hausdorff-distance instead of area overlap to compare trajectories. "
          "Default: False.")
    print("\t--hausdorffbias <bias>\t\tIf distance is less or equal this bias, two trajectories are considered "
          "identical. Should be a float. Default: 10.0.")
    print("")

    print("Options for finding endpoints:")
    print("\t--threshpen <pen>\t\t\tThe actual region (in percent, from right border) in which to look for the "
          "endpoints of the trajectories. Range: 0.00-1.00. Default: 0.70.")
    print("\t--ordermin <order>\t\t\tNumber of points (relative to width of image) used to compare for finding minima. "
          "Range: 0.00-1.00. Default: 0.05.")
    print("")

    print("Options for finding trajectories (gradient):")
    print("\t--maxiteration <number>\t\tMaximum iterations. Should be a positive integer. Default: 1000.")
    print("\t--gradient <factor>\t\t\tScaling factor for the gradient/focus field relative to max(width, height) "
          "of image. Range: 0.00-1.00. Default: 0.10.")
    print("\t--densityrad <radius>\t\tRadius of density field to calculate, 1 = 3x3 field, 2 = 5x5 field, etc. "
          "Has to be positive integer. Default: 1.")
    print("\t--maxattraction <a>\t\t\tAttraction of the inlets - the slope of the gradient will be "
          "influenced by this. Should be positive float. Default: 1.00.")
    print("")

    print("Options for finding trajectories (Dijkstra):")
    print("\t--useDijkstra\t\t\t\tUses path finding (Dijkstra) for trajectory finding")
    print("\t--densityrad <radius>\t\tRadius of density field to calculate, 1 = 3x3 field, 2 = 5x5 field, etc. "
          "Has to be positive integer. Default: 1.")
    print("\t--dijkstrabias <bias>\t\tBias for the density image. Should be an integer. Default: 50.")
    print("\t--everyotherpoint <points>\tOnly save every <points>. point. Otherwise a point for each pixel between "
          "start and end will be generated. Should be positive integer. Default: 10.")


# Step 0: Parse command-line arguments
# --------------------------------------------------------------------------------------------------------------------
# Debug mode
debugmode = False
debugcounter = 1

# Silent mode: Log only to file
silentmode = False

# Inputfile
inputfile = ""

# Channel to use (default = blank, which means: use complete image)
singlechannel = ""

# Several general parameters
zoneborder = 20             # The border of the zone will be overwritten with a black rectangle
threshbinary = 45           # Bias for binary picture (everything below 45 = black)
noendpoints = False         # Removes endpoints from trajectories
densityrad = 1              # Radius of density field to calculate, 1 = 3x3 field, 2 = 5x5 field, etc
useinlet = -1               # Uses only inlet # for evaluating paths; default (-1): use every inlet.
minpoints = 0               # Minimum points a trajectory must have to be saved

# Parameter for finding endpoints
threshpenetration = 0.70    # The actual region (in percent) in which to look for the endpoints of the trajectories
ordermin = 0.05             # Number of points (relative to width of image) used to compare for finding minima

# Parameter for finding trajectories (gradient)
maxiteration = 1000         # Maximum iterations for finding trajectories
gradientfactor = 0.10       # Scaling factor for the gradient/focus field
maxattraction = 1.0         # Attraction of the inlets - the slope will be influenced by this

# Parameter for finding trajectories (Dijkstra pathfinding)
useDijkstra = False         # Use Dijkstra pathfinding?
dijkstrabias = 50           # Bias for the density image
everyotherpoint = 10        # Only use every x. point (instead of all)

# Parameter for comparing trajectories
maxoverlap = 0.75           # Maximum overlap two trajectories can have before they are considered identical
useHausdorff = False        # Use Hausdorff-distance instead of overlap
hausdorffbias = 10.0        # If distance is less or equal this bias, two trajectories are considered identical



# No parameters given?
if len(sys.argv) == 1:
    printHelpPage()
    sys.exit(1)

# Try to find all the arguments
try:
    # All the options to recognize
    opts, args = getopt.getopt(sys.argv[1:], "", ["help", "input-file=", "debug", "silent", "channel=",
                                                  "zone-border=", "threshbin=", "useinlet=", "minpoints=",
                                                  "maxoverlap=", "threshpen=", "ordermin=", "noendpoints",
                                                  "maxiteration=", "gradient=", "densityrad=", "maxattraction=",
                                                  "useHausdorff", "hausdorffbias=",
                                                  "useDijkstra", "dijkstrabias=", "everyotherpoint="])

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
    elif opt == "--debug":
        debugmode = True
    elif opt == "--silent":
        silentmode = True
    elif opt == "--channel":
        if arg in ["blue", "green", "red"]:
            singlechannel = arg
        else:
            print("Channel has to be one of: blue, green, or red. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--zone-border":
        zoneborder = abs(int(arg))
    elif opt == "--threshbin":
        if 0 <= int(arg) <= 255:
            threshbinary = int(arg)
        else:
            print("Threshold should be an integer in the range of 0-255. '%s' was given." % str(arg))
    elif opt == "--useinlet":
        if 0 <= int(arg):
            useinlet = int(arg)
        else:
            print("Useinlet should be a positive integer. '%s' was given." % str(arg))
    elif opt == "--minpoints":
        if 0 <= int(arg):
            minpoints = int(arg)
        else:
            print("Minpoints should be a positive integer. '%s' was given." % str(arg))
    elif opt == "--maxoverlap":
        if 0.0 <= float(arg) <= 1.0:
            maxoverlap = float(arg)
        else:
            print("Maxoverlap should be a float in the range of 0.00-1.00. '%s' was given." % str(arg))
    elif opt == "--useHausdorff":
        useHausdorff = True
    elif opt == "--hausdorffbias":
        if 0.0 < float(arg):
            hausdorffbias = float(arg)
        else:
            print("Hausdorffbias should be a positive float. '%s' was given." % str(arg))
    elif opt == "--threshpen":
        if 0.0 <= float(arg) <= 1.0:
            threshpenetration = float(arg)
        else:
            print("Threshpen should be a float in the range of 0.00-1.00. '%s' was given." % str(arg))
    elif opt == "--ordermin":
        if 0.0 <= float(arg) <= 1.0:
            ordermin = float(arg)
        else:
            print("Ordermin should be a float in the range of 0.00-1.00. '%s' was given." % str(arg))
    elif opt == "--gradient":
        if 0.0 <= float(arg) <= 1.0:
            gradientfactor = float(arg)
        else:
            print("Gradient should be a float in the range of 0.00-1.00. '%s' was given." % str(arg))
    elif opt == "--maxattraction":
        if 0.0 < float(arg):
            maxattraction = float(arg)
        else:
            print("Maxattraction should be a positive float. '%s' was given." % str(arg))
    elif opt == "--maxiteration":
        if 0 < int(arg):
            maxiteration = int(arg)
        else:
            print("Maxiteration should be a positive integer. '%s' was given." % str(arg))
    elif opt == "--dijkstrabias":
        if 0 < int(arg):
            dijkstrabias = int(arg)
        else:
            print("Dijkstrabias should be an integer. '%s' was given." % str(arg))
    elif opt == "--everyotherpoint":
        if 0 < int(arg):
            everyotherpoint = int(arg)
        else:
            print("Everyotherpoint should be a positive integer. '%s' was given." % str(arg))
    elif opt == "--densityrad":
        if 0 < int(arg):
            densityrad = int(arg)
        else:
            print("Densityrad should be a positive integer. '%s' was given." % str(arg))
    elif opt == "--useDijkstra":
        useDijkstra = True
    elif opt == "--noendpoints":
        noendpoints = True

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
logger.info(u"####### Find trajectories #######")
logger.info(u"Trying to find trajectories on '%s'", inputfile)

# Debug mode?
if debugmode:
    ffe.enableDebugMode()

# Step 2: Read input image and data
# --------------------------------------------------------------------------------------------------------------------
# Read source image
inputimage = cv2.imread(inputfile, cv2.IMREAD_COLOR)

# Error?
if inputimage is None:
    logger.error(u"Input file could not be read")
    sys.exit(3)

# Shape of image
imageheight, imagewidth, imagebits = inputimage.shape

logger.info(u"Read input file")

# Use only one channel?
if singlechannel in ["blue", "green", "red"]:
    # Split original image into a dictionary
    imgchannels = {"blue": cv2.split(inputimage)[0],
                   "green": cv2.split(inputimage)[1],
                   "red": cv2.split(inputimage)[2]}
    # Create empty image
    inputimage = np.zeros((imageheight, imagewidth, 3), np.uint8)
    # Add channel
    inputimage[:, :, ["blue", "green", "red"].index(singlechannel)] = imgchannels[singlechannel]

    # Now add just the channel we want
    logger.info(u"Only %s channel is used for finding trajectories", singlechannel)

# Debug output and presentation
if debugmode:
    debugcounter = ffe.debugWriteImage(inputimage, debugcounter)
    cv2.imshow('Showcase', inputimage)

# Read FFE data
ffedata = ffe.loadDictionaryFromPng(inputfile)

logger.info(u"Extracting separation zone from image")

# Extracts the separation zone from the image
zoneimage, transmatrix = ffe.extractZoneFromImage(inputimage, ffedata["Inner separation zone"], zoneborder)

# Shape of zone image
zoneheight, zonewidth, zonebits = zoneimage.shape

logger.info(u"Extracted zone (%dx%d) from image (%dx%d)", zonewidth, zoneheight, imagewidth, imageheight)

# Debug output and presentation
if debugmode:
    debugcounter = ffe.debugWriteImage(zoneimage, debugcounter)
    cv2.imshow('Showcase', zoneimage)

# Step 3: Determine inverse(!) flow direction based on flow markers (if present)
# --------------------------------------------------------------------------------------------------------------------
# Standard inverse flow direction is on the horizontal line
flowdirection = (-1.00, 0.00)

# Flow markers present?
if "Flowmarkers" in ffedata:
    # Transform flow markers
    flowdirection = ffe.getInverseFlowDirection(ffedata["Flowmarkers"], transmatrix)

logger.info(u"Flow direction is (%.2f, %.2f)", flowdirection[0], flowdirection[1])

# Step 4: Calculate resolution
# --------------------------------------------------------------------------------------------------------------------
# This is the resolution for both axes in mm per pixel
resolution = (0.15, 0.15)

# Given in input file?
if "Physical zone dimensions" in ffedata:
    resolution = (float(ffedata["Physical zone dimensions"][0]) / float(zonewidth),
                  float(ffedata["Physical zone dimensions"][1]) / float(zoneheight))

logger.info(u"Resolution of image is %.2f x %.2f mm² per pixel²", resolution[0], resolution[1])

# Step 5: Get start points from inlets (if given) or construct one from flow markers/left border of zone
# --------------------------------------------------------------------------------------------------------------------
startpoints = []

# Inlet point(s) given?
if "Inlets" in ffedata:
    # We do not need the physical but the pixel represenation
    for inlet in ffedata["Inlets"]:
        startpoints.append((int(inlet[0] / resolution[0]),
                            int(inlet[1] / resolution[1]),
                            max(int(inlet[2] / ((resolution[1]+resolution[0])/2.0)), 1)))

    # Useinlet given and in the range of given inlets? Well, then only use this one start point
    if 0 <= useinlet < len(startpoints):
        logger.info(u"Only use inlet %d for evaluation.", useinlet)
        startpoints = [startpoints[useinlet]]

    logger.info(u"%d inlet(s) in data. Will be used as starting points.", len(startpoints))

# No starting points?
if len(startpoints) == 0:
    # Then we use the intersection of the flow line with the left zone border; the inlet has a width of 1 pixel
    if "Flowmarkers" in ffedata:
        startpoints.append((0, int(ffedata["Flowmarkers"][1][1] +
                                   ((0 - ffedata["Flowmarkers"][1][0])/flowdirection[0])*flowdirection[1]), 1))
    else:
        # This is the last draw, when nothing is provided: The middle point of the left border
        startpoints.append((0, int(zoneheight/2), 1))


# Step 6: Find trajectories for given channel
# --------------------------------------------------------------------------------------------------------------------
# Get channels
imgchannels = {"blue": cv2.split(zoneimage)[0],
               "green": cv2.split(zoneimage)[1],
               "red": cv2.split(zoneimage)[2]}

# Final list of trajectories
finaltrajectories = []

# For each channel
for channel in imgchannels:
    # If channel is empty, then skip
    if imgchannels[channel].max() == 0:
        continue

    # Debug output and presentation
    if debugmode:
        debugcounter = ffe.debugWriteImage(imgchannels[channel], debugcounter)
        cv2.imshow(channel, imgchannels[channel])

    logger.info(u"Search %s channel for trajectories", channel)

    # First, remove pixels under the threshhold
    ret, channelimage = cv2.threshold(imgchannels[channel], threshbinary, 255, cv2.THRESH_TOZERO)

    # Try to determine endpoints for channel with this blur image
    endpoints = ffe.findEndpointsOfTrajectories(channelimage, threshpenetration, ordermin)

    # No endpoints found?
    if len(endpoints) == 0:
        logger.warning(u"Could not find any endpoints for this channel. However, the channel itself is not empty.")
        continue

    # Get the trajectories from this channel using ...
    if useDijkstra:
        # ... Dijkstra pathfinding
        trajectories = ffe.getTrajectoriesFromImageDijkstra(channelimage, startpoints, endpoints, dijkstrabias,
                                                            everyotherpoint, densityrad)
    else:
        # ... gradient method
        trajectories = ffe.getTrajectoriesFromImage(channelimage, startpoints, endpoints,
                                                    flowdirection, maxiteration, gradientfactor,
                                                    densityrad, maxattraction)

    # Get the width of every point in the trajectories
    for index, trajectory in enumerate(trajectories):
        trajectories[index] = ffe.findWidthOfTrajectory(channelimage, trajectory, densityrad)

    # No trajectories found?
    if len(trajectories) == 0:
        logger.warning(u"Could not find any trajectories for this channel. However, found endpoints.")
        continue

    # Combine this trajectories with the trajectories already found (i.e. test for duplicates)
    finaltrajectories = ffe.combineTrajectories(finaltrajectories, trajectories, maxoverlap,
                                                useHausdorff, hausdorffbias)


logger.info(u"%d trajectories were found in file.", len(finaltrajectories))

# Remove endpoints?
if noendpoints:
    logger.info(u"Remove endpoints (--noendpoints given).")
    for i in xrange(len(finaltrajectories)):
        del finaltrajectories[i][-1]

# Filter all trajectories, which are less then minpoints
if minpoints > 0:
    logger.info(u"Remove all trajectories with less than %d points.", minpoints)

    # Create new list
    filteredtrajectories = []

    # Check trajectories
    for trajectory in finaltrajectories:
        if len(trajectory) >= minpoints:
            filteredtrajectories.append(trajectory)

    # Log
    logger.info(u"%d of %d trajectories were removed. %d remain.", len(finaltrajectories)-len(filteredtrajectories),
                len(finaltrajectories), len(filteredtrajectories))

    # Overwrite
    finaltrajectories = filteredtrajectories



# Debug output and presentation
if debugmode:
    # Create black and white copy
    zoneimagebw = cv2.cvtColor(zoneimage, cv2.COLOR_BGR2GRAY)
    zoneimagebw = cv2.cvtColor(zoneimagebw, cv2.COLOR_GRAY2BGR)

    # Delete blue and red channel
    zoneimagebw[:, :, 0] = 0
    zoneimagebw[:, :, 2] = 0

    # Reduce intensity of green channel
    zoneimagebw[:, :, 1] = cv2.subtract(zoneimagebw[:, :, 1], 100)

    # Draw trajectories on the image
    for line in finaltrajectories:
        for l in xrange(len(line) - 1):
            cv2.circle(zoneimagebw, (int(line[l][0]), int(line[l][1])), 2, (0, 215, 255), -5)
            cv2.arrowedLine(zoneimagebw, (int(line[l][0]), int(line[l][1])),
                            (int(line[l+1][0]), int(line[l+1][1])), (0, 215, 255), 2)

    # Draw inlets
    for pt in startpoints:
        cv2.circle(zoneimagebw, (pt[0], pt[1]), max(int(pt[2]/2), 1), (0, 0, 255), -5)

    debugcounter = ffe.debugWriteImage(zoneimagebw, debugcounter)
    cv2.imshow('Trajectories', zoneimagebw)


# Step 7: Save trajectories to file
# --------------------------------------------------------------------------------------------------------------------
# Here we put our trajectories into a dictionary and write it to the file;

# Create a dictionary
saveinfo = {}

# Save our trajectories
saveinfo.update({"Trajectories": finaltrajectories})

# Write to file
ffe.updateDictionaryOfPng(inputfile, saveinfo)

logger.info("Saved trajectories in file.")


# Last step: cleaning up
# --------------------------------------------------------------------------------------------------------------------
# Debugmode: wait for user input
if debugmode:
    cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()

# Final logging
logger.info(u"Finding trajectories ended on '%s'", inputfile)
logger.info(u"####### Find trajectories end #######")
