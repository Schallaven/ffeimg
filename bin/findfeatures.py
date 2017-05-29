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
import ConfigParser     # Reading/Writing config files
import ffe              # frequently-used function script
import time             # Time functions
import sys              # Sys functions
import logging          # Logging functions
import os               # OS functions
import png              # Raw PNG read/write functions
import shutil           # High level file operations (needed for copy at end)
import getopt           # Get and parse command-line arguments
import copy             # Make 1:1 copies of whole objects (numpy/opencv do not copy objects - they use references)

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
    print("\t--skip-flowmarkers:\t\t\tDo not look for flowmarkers.")
    print("\t--silent:\t\t\t\t\tDo not show anything in the console (except parameter errors).")
    print("")

    print("Options:")
    print("\t--input-file <file>\t\t\tInput file. <file> should be a PNG-file.")
    print("\t--channel <channel>\t\t\tUses only <channel> for feature finding. Default: All channels are used. "
          "<channel> can be blue, green, or red.")
    print("\t--thresh-binary ##\t\t\tThreshold for creating the binary picture. Default: 45. Range: 0-255.")
    print("\t--sepzone-area #.##\t\t\tThreshold for the area the separation zone must have at least given as fraction "
          "of the total pixels in the image. Default: 0.25. Range: 0.00-1.00.")
    print("\t--sepzone-ratio #.##\t\tRatio of area(inner contour)/area(outer contour) for detection of "
          "separation zone. Default: 0.8. Range: 0.00-1.00.")
    print("\t--sepzone-variance #.##\t\tVariance of above ratio. Default: 0.05. Range: 0.00-1.00.")
    print("\t--flowmarker-variance #.##\tVariance of marker area between the two flow markers. "
          "Default: 0.05. Range: 0.00-1.00.")
    print("\t--epsilon #.##\t\t\t\tEpsilon for refining the contours after detection. Default: 0.01. Range: 0.00-1.00")
    print("")


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

# Several thresh parameters
threshbinary = 45  # Bias for binary picture (everything above 45 = white)
thresharea = 0.25  # The separation zone takes up at least 25% of the whole area on the image.
threshratio = 0.8  # area(child)/area(parent)
variance_zone = 0.05  # The variance of the ratio of areas
variance_marker = 0.05  # Variance of marker area between the two flow markers
epsilon = 0.01  # For refining the contours of the separation zone boxes

# Skip looking for flow marker?
skip_flowmarkers = False

# No parameters given?
if len(sys.argv) == 1:
    printHelpPage()
    sys.exit(1)

# Try to find all the arguments
try:
    # All the options to recognize
    opts, args = getopt.getopt(sys.argv[1:], "", ["help", "input-file=", "thresh-binary=", "sepzone-area=",
                                                  "debug", "sepzone-ratio=", "sepzone-variance=",
                                                  "flowmarker-variance=", "skip-flowmarkers", "silent", "channel=",
                                                  "epsilon="])
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
    elif opt == "--skip-flowmarkers":
        skip_flowmarkers = True
    elif opt == "--silent":
        silentmode = True
    elif opt == "--thresh-binary":
        if 0 <= int(arg) < 256:
            threshbinary = int(arg)
        else:
            print("Threshbinary must be a integer in the range of 0 and 255. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--sepzone-area":
        if 0.00 <= float(arg) <= 1.00:
            thresharea = float(arg)
        else:
            print("Threshold area must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--sepzone-ratio":
        if 0.00 <= float(arg) <= 1.00:
            threshratio = float(arg)
        else:
            print("Threshold ratio must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--flowmarker-variance":
        if 0.00 <= float(arg) <= 1.00:
            variance_marker = float(arg)
        else:
            print("Variance_marker must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--sepzone-variance":
        if 0.00 <= float(arg) <= 1.00:
            variance_zone = float(arg)
        else:
            print("Variance_zone must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--channel":
        if arg in ["blue", "green", "red"]:
            singlechannel = arg
        else:
            print("Channel has to be one of: blue, green, or red. '%s' was given." % str(arg))
            sys.exit(0)
    elif opt == "--epsilon":
        if 0.00 <= float(arg) <= 1.00:
            epsilon = float(arg)
        else:
            print("Epsilon must be a float in the range of 0.00 and 1.00. '%s' was given." % str(arg))
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
logger.info(u"####### Find features #######")
logger.info(u"Trying to find features on '%s'", inputfile)

# Debug mode?
if debugmode:
    ffe.enableDebugMode()

# Step 2: Find contours
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
    logger.info(u"Only %s channel is used for finding features", singlechannel)

# Debug output and presentation
if debugmode:
    debugcounter = ffe.debugWriteImage(inputimage, debugcounter)
    cv2.imshow('Showcase', inputimage)

# Create gray image
grayimage = cv2.cvtColor(inputimage, cv2.COLOR_RGB2GRAY)

logger.info(u"Created gray image")

# Debug output and presentation
if debugmode:
    debugcounter = ffe.debugWriteImage(grayimage, debugcounter)
    cv2.imshow('Showcase', grayimage)

# Threshhold function
ret, binaryimage = cv2.threshold(grayimage, threshbinary, 255, cv2.THRESH_BINARY)

logger.info(u"Created binary image using %d as threshold", threshbinary)

# Debug output and presentation
if debugmode:
    debugcounter = ffe.debugWriteImage(binaryimage, debugcounter)
    cv2.imshow('Showcase', binaryimage)

# Find contours
contour_bw_img, contours, hierarchy = cv2.findContours(binaryimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

logger.info(u"Searching for contours")

# Debug output and presentation
if debugmode:
    # Make a copy for the contour image
    cntimage = copy.copy(inputimage)
    cv2.drawContours(cntimage, contours, -1, (0, 0, 255), 1)
    debugcounter = ffe.debugWriteImage(cntimage, debugcounter)
    cv2.imshow('Showcase', cntimage)

# No contours?
if len(contours) == 0:
    logger.error(u"Did not find any contours")
    sys.exit(3)

logger.info(u"Found %d contours", len(contours))

# Calculate the moments / center-points of the contours
moments = [0] * len(contours)
centers = [0, 0] * len(contours)

# Fill arrays
for i in xrange(len(contours)):
    moments[i] = cv2.moments(contours[i])
    if moments[i]['m00'] == 0:
        centers[i] = (0, 0)
    else:
        centers[i] = (int(round(moments[i]['m10'] / moments[i]['m00'], 0)),
                      int(round(moments[i]['m01'] / moments[i]['m00'], 0)))

logger.info(u"Calculated center-points of contours")

# Step 3: Find separation zone
# --------------------------------------------------------------------------------------------------------------------
# We need to find a top-level contour, which has another contour in it. The ratio is approx. 0.8. We also assume that
# the top-level contour takes up most of the space of the picture. At least the following threshold:
threshpixels = int(thresharea * imagewidth * imageheight)

logger.info(u"Trying to find top-level contour with at least %d pixels in size", threshpixels)

# Find boundaries
parentindex, childindex = ffe.findSeparationZoneBoundaries(contours, hierarchy, threshpixels, threshratio,
                                                           variance_zone)

# Nothing found?
if parentindex == -1 or childindex == -1:
    logger.error(u"No contour was found, which matches the criteria for the separation zone")
    sys.exit(4)

logger.info(u"Found a matching contour pair (%d, %d) for the separation zone", parentindex, childindex)

# Approximation of contour shape = Refining (Douglas-Peucker algorithm)
logger.info(u"Refining the contours.")

# Refining
contours[parentindex] = cv2.approxPolyDP(contours[parentindex],
                                         epsilon * cv2.arcLength(contours[parentindex], True), True)

contours[childindex] = cv2.approxPolyDP(contours[childindex],
                                        epsilon * cv2.arcLength(contours[childindex], True), True)

# Get boxes for parent and child
boxparent = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[parentindex])))
boxchild = np.int0(cv2.boxPoints(cv2.minAreaRect(contours[childindex])))

# Debug output and presentation
if debugmode:
    # Draw contour for parent
    cv2.drawContours(inputimage, [boxparent], 0, (0, 215, 255), 2)
    # Draw contour for child
    cv2.drawContours(inputimage, [boxchild], 0, (0, 215, 255), 2)
    # Draw mid-point
    cv2.circle(inputimage, centers[childindex], 5, (0, 215, 255), -5)

    debugcounter = ffe.debugWriteImage(inputimage, debugcounter)
    cv2.imshow('Showcase', inputimage)

# Sort the coordinates, so that the order will be top-left, top-right, bottom-left, bottom-right
boxparent = ffe.sortCoordinates(boxparent, (imagewidth, imageheight))
boxchild = ffe.sortCoordinates(boxchild, (imagewidth, imageheight))


# Step 4: Find flow markers
# --------------------------------------------------------------------------------------------------------------------
# The idea here is to find two almost equal contours, which are mirrored by the center point of boxchild. The contours
# are outside of the parentbox

# Define this variable
flowmarkerindex = (-1, -1)

# Skip?
if not skip_flowmarkers:
    logger.info(u"Trying to find the flow markers.")

    # Find flow markers
    flowmarkerindex = ffe.findFlowMarkers(contours, (parentindex, childindex), boxparent, variance_marker)

    if flowmarkerindex[0] == -1 or flowmarkerindex[1] == -1:
        logger.error(u"Did not find any flowmarker pair.")
        sys.exit(5)

    # Debug output and presentation
    if debugmode:
        # Make a copy for the contour image
        cv2.circle(inputimage, centers[flowmarkerindex[0]], 5, (0, 255, 0), -5)
        cv2.circle(inputimage, centers[flowmarkerindex[1]], 5, (0, 255, 0), -5)
        cv2.line(inputimage, centers[flowmarkerindex[0]], centers[flowmarkerindex[1]], (0, 255, 0), 2)

        debugcounter = ffe.debugWriteImage(inputimage, debugcounter)
        cv2.imshow('Showcase', inputimage)

else:
    logger.info(u"Did not look for flow markers (--skip-flowmarkers given).")

# Step 5: Save data in png file
# --------------------------------------------------------------------------------------------------------------------
# Here we put our results into a dictionary and write it to the file; we want to make sure that the coordinates are
# always tuples or list of tuples

# Create a dictionary
saveinfo = {}

# Insert outer box of separation zone (parentbox)
saveinfo.update({"Outer separation zone":
                     [tuple(boxparent[0]), tuple(boxparent[1]), tuple(boxparent[2]), tuple(boxparent[3])]})

# Insert inner box of separation zone (childbox)
saveinfo.update({"Inner separation zone":
                     [tuple(boxchild[0]), tuple(boxchild[1]), tuple(boxchild[2]), tuple(boxchild[3])]})

# Insert flow markers (if not skipped)
if not skip_flowmarkers:
    # Sort the flow markers from left to right (i.e. by distance to the middle point on the very left of the image)
    flowmarkerlist = ffe.sortCoordinatesByDistanceToPoint(
        np.array([tuple(centers[flowmarkerindex[0]]), tuple(centers[flowmarkerindex[1]])]), (0, imageheight / 2))
    # Save
    saveinfo.update({"Flowmarkers": [tuple(flowmarkerlist[0]), tuple(flowmarkerlist[1])]})

# Write to file
ffe.updateDictionaryOfPng(inputfile, saveinfo)

logger.info("Saved coordinates in file.")

# Last step: cleaning up
# --------------------------------------------------------------------------------------------------------------------
# Debugmode: wait for user input
if debugmode:
    cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()

# Final logging
logger.info(u"Finding features ended on '%s'", inputfile)
logger.info(u"####### Find features end #######")
