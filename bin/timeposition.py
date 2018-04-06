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
# timeposition.py (example)
#
# This quick&dirty example script extracts the position of the front as a function of time for the measured flowchannel
# profiles and saves them as flowch1-5.csv
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
import scipy.stats                  # For the linear regression
import os                           # Some operating system functions


# Input directories
inputdirs = ['flowch' + str(i+1) for i in xrange(5)]

# For every directory...
for channelindex, channel in enumerate(inputdirs):
    # Status
    print("Analyzing %s ..." % channel)

    # Output table for results from this channel
    outputchannel = []

    # For every FFE file in this directory
    for file in os.listdir(channel):

        # Only FFE.png files:
        if not file.endswith('FFE.png'):
            continue

        # Read source image
        inputimage = cv2.imread(channel+'/'+file, cv2.IMREAD_COLOR)

        # Error?
        if inputimage is None:
            print("Input file could not be read")
            sys.exit(3)

        # Shape of image
        imageheight, imagewidth, imagebits = inputimage.shape

        # Read data
        ffedata = ffe.loadDictionaryFromPng(channel+'/'+file)

        # Are there trajectories in file's data?
        if "Inner separation zone" not in ffedata:
            print("No inner separation zone in file %s" % (channel+'/'+file))
            continue

        # No trajectory data? or No inlet data? Skip
        if "Trajectories" not in ffedata or "Inlets" not in ffedata:
            continue

        # No trajectories?
        if len(ffedata["Trajectories"]) == 0:
            continue

        # Not enough inlets? (should be 5)
        if len(ffedata["Inlets"]) != 5:
            continue

        # Extracts the separation zone from the image
        zoneimage, transmatrix = ffe.extractZoneFromImage(inputimage, ffedata["Inner separation zone"], 10)

        # Shape of zone image
        zoneheight, zonewidth, zonebits = zoneimage.shape

        # This is the resolution for both axes in mm per pixel
        resolution = (0.15, 0.15)

        # Given in input file?
        if "Physical zone dimensions" in ffedata:
            resolution = (float(ffedata["Physical zone dimensions"][0]) / float(zonewidth),
                          float(ffedata["Physical zone dimensions"][1]) / float(zoneheight))

        # Convert inlets to array
        inlets = np.array([(item[0], item[1]) for item in ffedata["Inlets"]])

        # Calculate the inlet we want to observe; measurements were done from inlet 1-5; Inlets have all the same x;
        # round and convert to int
        injectinlet = (int(round(inlets[0][0])), int(round(inlets[4][1]-channelindex*(inlets[4][1]-inlets[3][1]))))

        #print("Inject ", injectinlet)

        # Let the origin be the first inlet if exists (origin is in real dimensions, i.e. mm!)
        origin = (0, 0)

        # First sort the points by distance from the center of the left side (0, zoneheight/2)
        inlets = ffe.sortCoordinatesByDistanceToPoint(inlets, (0, int(ffedata["Physical zone dimensions"][1] / 2)))

        # Calculate new origin from nearest point (first point in array)
        origin = (inlets[0][0], inlets[0][1])

        # Check each trajectory
        for trajectory in ffedata["Trajectories"]:
            # Get first point in real coordinates; round and convert to int
            start = (int(round(trajectory[0][0]*resolution[0])),
                     int(round(trajectory[0][1]*resolution[1])))

            # Check if start is injectinlet, if not continue (we only want to consider the trajectories, which are
            # connected to the corresponding inlets); We use the rounded inletwidth as boundary
            if np.hypot(start[0]-injectinlet[0], start[1]-injectinlet[1]) > int(round(ffedata["Inlets"][0][2])):
                continue

            # Collect data for saving to file: relative time, coordinates of last point, width of last point
            time = ffedata["Snapshot time"]
            x, y, w = trajectory[-1][0]*resolution[0]-origin[0], \
                      trajectory[-1][1]*resolution[1]-origin[1], \
                      trajectory[-1][2]*resolution[1]  # Just use resolution of y and no origin for width

            # Append results to list
            outputchannel.append((time, x, y, w))

            #print("%.2f: %.2f, %.2f, %2.f" % (time, x, y, w))

    # Convert output to structured numpy array
    outputchannel = np.array(outputchannel, dtype=[(channel + ' time', 'float64'), (channel + ' x', 'float64'),
                                                   (channel + ' y', 'float64'), (channel + ' width', 'float64')])

    # Save this numpy array
    np.savetxt(channel+'.csv', outputchannel, delimiter='\t', header="\t".join(outputchannel.dtype.names))








