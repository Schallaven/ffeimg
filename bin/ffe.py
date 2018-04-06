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

"""
This module defines commonly used functions (ffe = frequently-used function for evaluation) for the imaging system.
Absolutely no pun intended.
Logs almost everything to a (non-configured) logger called 'FFE'.

Compatibility
-------------
`ffe` has been tested and developed on Python 2.7.

Requirements
-------------
The module and user scripts use mainly standard library functions, which come with Python 2.7. Common packages such as
 `Matplotlib`, `Numpy`, `Scipy` should already be present in most Python installations. Additionally, `pyPNG` and
 `OpenCV` are required.

All non-standard packages, except `OpenCV`, can be installed at once using PIP and the provided `requirements.txt`
 file:

`pip install -r requirements.txt`

For installing `OpenCV` (Open Source Computer Vision) please check the [respective documentation](http://opencv.org/).


Installation
------------
In order to install the module and the user scripts, copy the contents of the package to a directory of your choice.
 Simply run setup.py in the directory of the package to register the `ffe` module with python:

`python setup.py install`

Then, add the directory to your PATH (the executable search path). For Windows, you can use the `setenv.bat`-batch file
 provided with the package files.


Contributing
------------
`ffe` [is on GitHub](https://github.com/blablabla/ffe). Pull requests and bug reports are welcome.

Furthermore, we welcome and invite persons and institutes, in particular those in the field of Free Flow
 Electrophoresis, who want to actively contribute to the development of `ffe`. Please contact the authors for more
 information.


License
-------
`ffe` is published under the MIT license.


Documentation
-------------
Documentation for `ffe` module was automatically generated using [`pdoc`](https://github.com/BurntSushi/pdoc).

"""

# Import modules
import cv2                          # OpenCV
import numpy as np                  # Numpy - You always need this.
from numpy.lib.utils import source
import ConfigParser                 # Reading/Writing config files
import logging                      # Logging of some sort
import png                          # Raw PNG read/write functions
import cPickle                      # For serializing dictionaries (and other things)
import math                         # Math! You didn't think we could go without any math, right?
import matplotlib.pyplot as plt     # For plotting intermediate graphs (debug)
import scipy                        # Scientific things!
import scipy.signal                 # Unfortunately, every submodule has to be imported explicitely for scipy :(
import scipy.interpolate
import operator                     # Operator tools
import itertools                    # Iteration tools
import copy                         # Make 1:1 copies of whole objects (numpy/opencv do not copy objects)
import heapq                        # Fast implementation of a priority queue
import time                         # Time functions
import warnings                     # Scipy module and some others use this for generating warnings
import os                           # Some operating system functions


# Create but do not configure logger
logger = logging.getLogger('FFE')
"""This variable holds the logger named `FFE`. Most subroutines in this module will forward info and debug output to
 this logger. It needs to be configured by the main program (see
 [`logging` module manual](https://docs.python.org/2/library/logging.html))."""

# This is for debug purposes
debugmode = False
"""Debug flag (Default: `False`). Users can use `ffe.enableDebugMode` and `ffe.disableDebugMode` to control this flag.
 When enabled many functions will provide additional debug data in form of text (send to `ffe.logger`) or graphical
 output (<i>e.g.</i> plot windows). This limits the automation of evaluation and should only be used for debugging
 purposes."""


# Function: Enable debug mode
def enableDebugMode():
    """Enables debug mode. Only sets `ffe.debugmode` to `True` at the moment."""
    global debugmode
    debugmode = True


# Function: Disable debug mode
def disableDebugMode():
    """Disables debug mode. Only sets `ffe.debugmode` to `False` at the moment."""
    global debugmode
    debugmode = False


# Function: Returns the camera ID from 'hardware.ini' or zero if not found.
def getCameraID():
    """Get the camera ID. Unfortunately OpenCV has no means at the moment to get a list of camera devices and/or IDs
    to make this process more comfortable. For now, the user provides the ID by a file 'hardware.ini'. If not found zero is
    used as standard ID.
    """

    # Default ID = 0
    camid = 0

    # Does 'hardware.ini' exist (in the current diectory)
    if os.path.isfile('hardware.ini'):
        # Read the content of that file and convert it to an integer
        with open('hardware.ini', 'r') as hardwarefile:
            camid = int(hardwarefile.read())

    # Returns a usable camera id (=> 0)
    return abs(camid)

# Function: Read and apply the camera settings; returns a dictionary for the recording settings
def applyCameraSettings(camera):
    """Read and apply the camera settings from `camera.ini`. `camera` is a camera object from `OpenCV`
    (`cv2.VideoCapture()`). Returns a dictionary with the recording settings.
    """
    # Read camera settings
    Config = ConfigParser.ConfigParser()
    Config.read("camera.ini")

    # Apply general camera settings
    if Config.has_section("Camera"):
        # General settings
        camera.set(cv2.CAP_PROP_BRIGHTNESS, Config.getfloat("Camera", "Brightness"))  # 0-255
        camera.set(cv2.CAP_PROP_CONTRAST, Config.getfloat("Camera", "Contrast"))  # 0-255
        camera.set(cv2.CAP_PROP_SATURATION, Config.getfloat("Camera", "Saturation"))  # 0-255
        camera.set(cv2.CAP_PROP_SHARPNESS, Config.getfloat("Camera", "Sharpness"))  # 0-255
        camera.set(cv2.CAP_PROP_GAIN, Config.getfloat("Camera", "Gain"))  # 0-100
        camera.set(cv2.CAP_PROP_HUE, Config.getfloat("Camera", "Hue"))  # fixed on 13?
        camera.set(cv2.CAP_PROP_BACKLIGHT, Config.getfloat("Camera", "Backlight"))
        # Frame size
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.getfloat("Camera", "Framewidth"))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.getfloat("Camera", "Frameheight"))
        # Exposure
        camera.set(cv2.CAP_PROP_EXPOSURE, Config.getfloat("Camera", "Exposure"))

    # Read the recording settings and store in dictionary
    records = {}
    if Config.has_section("Recording"):
        # Flipping
        records["doflipping"] = False
        if Config.getint("Recording", "flipping") > 0:
            flip = Config.getint("Recording", "flipping") - 2
            if flip == 2:
                flip = -1
            records["doflipping"] = True
            records["flip"] = flip
        # Integrate over this time [in seconds]
        records["integratetime"] = Config.getfloat("Recording", "integratetime")
        # Which channels?
        records["allchannels"] = Config.get("Recording", "channels")
        channels = records["allchannels"].split(",")
        for channel in channels:
            records[channel] = True

    # Return the dictionary
    return records


# Function: Dumps camera properties to the console
def dumpCameraPropsToConsole(camera):
    """Simply dumps camera properties to the console. It is meant mainly for debug purposes, in case no logger is
    available. `camera` is a camera object from `OpenCV` (`cv2.VideoCapture()`).
    """
    print(u"Width x Height: %.1f x %.1f" % (camera.get(cv2.CAP_PROP_FRAME_WIDTH),camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(u"Brightness: %.1f" % camera.get(cv2.CAP_PROP_BRIGHTNESS))
    print(u"Contrast: %.1f" % camera.get(cv2.CAP_PROP_CONTRAST))
    print(u"Saturation: %.1f" % camera.get(cv2.CAP_PROP_SATURATION))
    print(u"Sharpness: %.1f" % camera.get(cv2.CAP_PROP_SHARPNESS))
    print(u"Gain: %.1f" % camera.get(cv2.CAP_PROP_GAIN))
    print(u"Hue: %.1f" % camera.get(cv2.CAP_PROP_HUE))
    print(u"Backlight: %.1f" % camera.get(cv2.CAP_PROP_BACKLIGHT))
    print(u"Exposure: %.1f" % (camera.get(cv2.CAP_PROP_EXPOSURE)))


# Function: Logs camera properties
def logCameraProps(camera):
    """Logs all camera settings to a logger (`FFE`). `camera` is a camera object from `OpenCV` (`cv2.VideoCapture()`).
    """
    logger.info(u"Camera\tResolution: %.1f × %.1f",camera.get(cv2.CAP_PROP_FRAME_WIDTH),camera.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    logger.info(u"Camera\tBrightness: %.1f", camera.get(cv2.CAP_PROP_BRIGHTNESS))
    logger.info(u"Camera\tContrast:   %.1f", camera.get(cv2.CAP_PROP_CONTRAST))
    logger.info(u"Camera\tSaturation: %.1f", camera.get(cv2.CAP_PROP_SATURATION))
    logger.info(u"Camera\tSharpness:  %.1f", camera.get(cv2.CAP_PROP_SHARPNESS))
    logger.info(u"Camera\tGain:       %.1f", camera.get(cv2.CAP_PROP_GAIN))
    logger.info(u"Camera\tHue:        %.1f", camera.get(cv2.CAP_PROP_HUE))
    logger.info(u"Camera\tBacklight:  %.1f", camera.get(cv2.CAP_PROP_BACKLIGHT))
    logger.info(u"Camera\tExposure:   %.1f", (camera.get(cv2.CAP_PROP_EXPOSURE)))


# Function: logs the recording settings
def logRecordingSettings(records):
    """Logs recording settings. `records` is the dictionary returned by `ffe.applyCameraSettings`.
    """
    flipped = u"not flipped"
    if records["doflipping"]:
        flipped = [u"flipped horizontally and vertically", u"flipped vertically only", u"flipped horizontally only"][records["flip"]+1]
    logger.info(u"Recording channels (%s).", records["allchannels"])
    logger.info(u"Images will be integrated over %.1fs. Images are %s", float(records["integratetime"]), flipped)


# Function: Read output settings and return as dictionary
def getOutputSettings():
    """Reads output settings from `output.ini` and returns them as dictionary. Will replace an empty experiment name
    and by `Random experiment` if necessary.
    """
    # Read from config file
    Config = ConfigParser.ConfigParser()
    Config.read("output.ini")

    # Put into dictionary
    settings = dict(Config.items("Output"))

    # No name given?
    if len(settings["name"]) == 0:
        settings["name"] = "Random experiment"

    # Return
    return settings


# Function: Logs output settings
def logOutputSettings(settings):
    """Logs output settings. `settings` is the dictionary returned by `ffe.getOutputSettings`.
    """
    logger.info(u"Output name: %s", settings["name"])
    logger.info(u"Output rootdir: %s", settings["rootdir"])
    logger.info(u"Output data: %s", settings["dataline1"])
    logger.info(u"Output data: %s", settings["dataline2"])
    logger.info(u"Output video: %s with %d FPS", settings["videocodec"], int(settings["videofps"]))
    logger.info(u"Output a snapshot every %d frames", int(settings["snapshots"]))
    if int(settings["combinechannels"]) == 1:
        logger.info(u"Recording will combine all input channels to a single green output channel in "
                    u"both the video and the images.")
    if int(settings["scaleindividually"]) == 1:
        logger.info(u"Recording will scale the channels individually.")
    else:
        logger.info(u"Recording will scale the channels together.")
    if int(settings["ignorebackground"]) == 1:
        logger.info(u"Background will not be subtracted from the frames.")


# Function: Scales a n-bit frame to 0-255 and returns it in 8bit form
def rescaleFrameTo8bit(frame):
    """Scales a n-bit image to 0-255 and returns it in 8bit form. `frame` is the input image given as `numpy.ndarray`.
    Returns a `numpy.ndarray` or `None` if there is an error.
    """
    # Is frame not a numpy array?
    if type(frame) is not np.ndarray:
        logger.error(u"rescale32bitFrameTo8bit(frame): frame is not a numpy array")
        return None

    # Catch division by zero: Whole frame is black, still convert to 8bit
    if frame.max() == 0:
        return frame.astype(np.uint8)

    # Rescale and return (There is also a cv2.convertScaleAbs function)
    return (np.multiply(frame, 255) / (frame.max())).astype(np.uint8)


# Function: Adds chunks to a PNG (before IEND)
def addDataChunksToPng(filename, chunks):
    """Adds chunks to a PNG file (right before the `IEND`-chunk). `filename` is the filename of the PNG file to be
    modified and `chunks` is a list of chunks to be added. Every entry in the list should have the format of
    `["CHNK", "Content"]`. See `ffe.createChunksFromDictionary` for creating a chunk list from a dictionary.
    """
    # Is filename a string?
    if type(filename) is not str:
        logger.error(u"addDataChunkToPng(filename, chunks): filename is not a string")
        return None

    # Is data a dictionary?
    if type(chunks) is not list:
        logger.error(u"addDataChunkToPng(filename, chunks): chunks is not a list")
        return None

    # List not 2D or empty
    if len(chunks) == 0 or not len(chunks[0]) == 2:
        logger.error(u"addDataChunkToPng(filename, chunks): chunks is not a 2D list or empty")
        return None

    # Open the file
    file = png.Reader(filename)

    # Get list of chunks in file
    chunklist = list(file.chunks())

    # Remove last item (IEND)
    del chunklist[-1]

    # Now add the above chunks to the list
    chunklist.extend(chunks)

    # Add IEND
    chunklist.append(["IEND", ""])

    # Now write back the chunks
    with open(filename, "wb") as file:
        png.write_chunks(file, chunklist)


# Function: Creates tEXt and dfFe chunks from a dictionary
def createChunksFromDictionary(data):
    """Creates `tEXt` and `dfFe` chunks from a dictionary. `data` is the dictionary of data to be transformed
    in `tEXt` and `dfFe` chunks. Returns the chunks as list, which can be subsequently given to
    `ffe.addDataChunksToPng`.
    """
    # Is data a dictionary?
    if type(data) is not dict:
        logger.error(u"createChunksFromDictionary(data): data is not a dictionary")
        return None

    # Chunks
    chunks = []

    # For dfFe we just serialize the dictionary
    chunks.append(["dfFe", cPickle.dumps(data)])

    # We also want to put them a tEXt into the file, so that PNG reader can list them
    for key in data:
        # Truncate key to 79 characters (if needed)
        keytext = key[:79]
        # Append chunk
        chunks.append(["tEXt", "%s\x00%s" % (keytext, data[key])])

    # Return chunks
    return chunks


# Function: Loads the dictionary from the dfFe chunk
def loadDictionaryFromPng(filename):
    """Loads the dictionary from a `dfFe` chunk of a PNG file given by `filename`. Returns the data from the `dfFe`
    chunk as dictionary or `None` if an error occurs.
    """
    # Is filename a string?
    if type(filename) is not str:
        logger.error(u"loadDictionaryFromPng(filename): filename is not a string")
        return None

    # Open the file
    file = png.Reader(filename)

    # Get list of chunks in file
    chunklist = list(file.chunks())

    # Create empty dictionray
    data = {}

    # Find the dfFe chunk(s)
    for c in chunklist:
        if str(c[0]) == "dfFe":
            # Append to dictionary (overwrites data with the same keys)
            data.update(cPickle.loads(str(c[1])))

    # Return dictionary
    return data


# Function: Updates the dictionary in the dfFe chunk of a PNG
def updateDictionaryOfPng(filename, updatedata):
    """Updates the dictionary in a `dfFe` chunk of a PNG file. `filename` is the PNG file to update and `updatedata`
    is the data in form of a dictionary, which will be added or updated to the data already present in the PNG file.
    """
    # Is filename a string?
    if type(filename) is not str:
        logger.error(u"updateDictionaryOfPng(filename, updatedata): filename is not a string")
        return

    # Is data a dictionary?
    if type(updatedata) is not dict:
        logger.error(u"updateDictionaryOfPng(filename, updatedata): data is not a dictionary")
        return

    # Open the file
    pngfile = png.Reader(filename)

    # Get list of chunks in file
    chunklist = list(pngfile.chunks())

    # Remove last item (IEND)
    del chunklist[-1]

    # Create empty dictionray
    data = {}

    # Find the dfFe and tEXt chunk(s)
    for c in chunklist:
        if str(c[0]) == "dfFe":
            # Append to dictionary (overwrites data with the same keys)
            data.update(cPickle.loads(str(c[1])))

    # Now update the dictionary
    data.update(updatedata)

    # Create new dfFe and tEXt chunks
    datachunks = createChunksFromDictionary(data)

    # Create a new chunklist without the old dfFe
    chunklist = [c for c in chunklist if not str(c[0]) == "dfFe" and not str(c[0]) == "tEXt"]

    # Add chunks from above
    chunklist.extend(datachunks)

    # Add IEND
    chunklist.append(["IEND", ""])

    # Now write back the chunks
    with open(filename, "wb") as file:
        png.write_chunks(file, chunklist)


# Function: Replaces the dictionary in the dfFe chunk of a PNG
def replaceDictionaryOfPng(filename, replacedata):
    """Replaces the `dfFe` chunk of a PNG file with new data. `filename` is the PNG file and `replacedata` is
    the data, which will replace the data in the PNG file or just added if no `dfFe` chunk is present.
    """
    # Is filename a string?
    if type(filename) is not str:
        logger.error(u"replaceDictionaryOfPng(filename, updatedata): filename is not a string")
        return

    # Is data a dictionary?
    if type(replacedata) is not dict:
        logger.error(u"replaceDictionaryOfPng(filename, updatedata): data is not a dictionary")
        return

    # Open the file
    pngfile = png.Reader(filename)

    # Get list of chunks in file
    chunklist = list(pngfile.chunks())

    # Remove last item (IEND)
    del chunklist[-1]

    # Create new dfFe and tEXt chunks
    datachunks = createChunksFromDictionary(replacedata)

    # Create a new chunklist without the old dfFe
    chunklist = [c for c in chunklist if not str(c[0]) == "dfFe" and not str(c[0]) == "tEXt"]

    # Add chunks from above
    chunklist.extend(datachunks)

    # Add IEND
    chunklist.append(["IEND", ""])

    # Now write back the chunks
    with open(filename, "wb") as file:
        png.write_chunks(file, chunklist)


# Function: Prepares an overlay image based on the some filedata
def prepareOverlayImage((width, height), color, data):
    """Prepares the overlay image for user presentation. Is used to render the live view and the frames in the video
    files. `width` and `height` is the requested size of the overlay image. `color` is the color given in BGR used
    for text rendering. `data` is a dictionary of information, which will be printed on the overlay image (`Timestamp`,
    `Snapshot time`, `Dataline 1`, and `Dataline 2`).

    Returns an overlay image in form of a numpy array.
    """
    # Is data a dictionary?
    if type(data) is not dict:
        logger.error(u"prepareOverlayImage(..., data): data is not a dictionary")
        return None

    # Width/Height should be ints
    if type(width) is not int or type(height) is not int:
        logger.error(u"prepareOverlayImage((width, height), ...): width and/or data are no integers")
        return None

    # Create empty image
    overlayimage = np.zeros((height, width, 3), np.uint8)

    # Select font
    fontface = cv2.FONT_HERSHEY_SIMPLEX

    # Adding timestamp
    textsize, baseline = cv2.getTextSize(data["Timestamp"], fontface, 0.5, 1)
    textwidth, textheight = textsize
    cv2.putText(overlayimage, data["Timestamp"], (10, textheight + 10), fontface, 0.5, color, 1, 8, False)

    # Adding relative time
    timedf = data["Snapshot time"]
    relativetime = 'time (min:sec): %02d:%05.2f' % (int(timedf / 60), timedf - int(timedf / 60) * 60.0)
    textsize, baseline = cv2.getTextSize(relativetime, fontface, 0.5, 1)
    textwidth, textheight = textsize
    cv2.putText(overlayimage, relativetime, (width - 10 - textwidth, textheight + 10), fontface, 0.5, color, 1, 8, False)

    # Adding relative time (in seconds)
    relativetime = 'time (sec): %.2f' % timedf
    textsize, baseline = cv2.getTextSize(relativetime, fontface, 0.5, 1)
    textwidth, textheight = textsize
    cv2.putText(overlayimage, relativetime, (width - 10 - textwidth, textheight + 37), fontface, 0.5, color, 1, 8, False)

    # Dataline1
    textsize, baseline = cv2.getTextSize(data["Dataline 1"], fontface, 0.5, 1)
    textwidth, textheight = textsize
    cv2.putText(overlayimage, data["Dataline 1"], (10, textheight + 37), fontface, 0.5, color, 1, 8, False)

    # Dataline2
    textsize, baseline = cv2.getTextSize(data["Dataline 2"], fontface, 0.5, 1)
    textwidth, textheight = textsize
    cv2.putText(overlayimage, data["Dataline 2"], (10, textheight + 64), fontface, 0.5, color, 1, 8, False)

    # Return image
    return overlayimage


# Function: Tries to find the two boundary contours for the separation zone
def findSeparationZoneBoundaries(contours, hierarchy, threshold, threshratio, variance):
    """Tries to find the inner and outer boundary contours for the separation zone. `contours` and `hierarchy` are the
    respective return values from `cv2.findContours`.

    As mentioned in the main text, the outer boundary has to be at least `threshold` pixels in size (area) and the
    ratio of inner contour (area)/outer contour (area) has to be in the range of `(threshratio-variance)` to
    `(threshratio+variance)`.

    Returns the indices (`int, int`) of the outer boundary and inner boundary in `contours`. Returns `-1, -1` if
    no contour was found, which matches the requirements, or an error occurred.
    """
    # Contours and hierarchy should match
    if len(contours) != len(hierarchy[0]):
        logger.error(u"Contours and hierarchy do not match.")
        return -1, -1

    # Structure of hierarchy: [(next, prev, child, parent)]; Do not know why there is a extra dimension - the
    # documentation tells other stories...

    # Process contours
    for i in xrange(len(contours)):
        # Has a parent? Then continue
        if hierarchy[0][i][3] != -1:
            logger.debug(u"Contour %d has a parent", i)
            continue

        # Has no children? Then continue
        if hierarchy[0][i][2] == -1:
            logger.debug(u"Contour %d has no children", i)
            continue

        # Area of parent
        areaparent = float(cv2.contourArea(contours[i]))

        if areaparent == 0.0:
            areaparent = 1.0

        # If area of parent is less then our theshold, continue
        if areaparent < threshold:
            continue

        # Found a top-level contour with children
        logger.info(u"Contour %d is a candidate for the separation zone. Area of contour: %.2f", i, areaparent)

        # Now process all children
        for j in xrange(len(contours)):
            # Parent is our top-level contour?
            if hierarchy[0][j][3] != i:
                continue

            # Area of child contour
            areachild = float(cv2.contourArea(contours[j]))

            # If area of child is less then our theshold, continue
            if areachild < threshold:
                continue

            # Ratio (rounded)
            ratio = round(areachild / areaparent, 1)

            logger.info(u"Found child %d with area of %.2f. Ratio: %.2f", j, areachild, ratio)

            # Found a candidate!
            if (threshratio-variance) <= ratio <= (threshratio+variance):
                logger.info(u"Child is in ratio range: %.2f <= %.2f <= %.2f",
                            threshratio-variance, ratio, threshratio+variance)
                # Save results and exit!
                return i, j

    # Nothing found?
    return -1, -1


# Function: Debug output as debugNNN.png. Returns counter + 1. So, it can be used as
#           debugcounter = ffe.debugWriteImage(image, debugcounter)
def debugWriteImage(image, counter):
    """ Writes an `image` (numpy array) to `debugNNN.png` (`NNN` replaced by counter) for debug purposes. Also, logs
    the output to the file. Returns `counter+1`. This function is meant to be used as

    `debugcounter = ffe.debugWriteImage(image, debugcounter)`.

    This is used by <i>e.g.</i> `findfeatures.py` in debug mode to output the intermediate images during the various
    evaluation steps.
    """
    # Counter should be an integer
    if type(counter) is not int:
        logger.error(u"debugWriteImage(image, counter): counter is no integer")
        return -1

    # Image should be a numpy array
    if type(image) is not np.ndarray:
        logger.error(u"debugWriteImage(image, counter): image is no numpy array")
        return -1

    # Write to filename
    cv2.imwrite("debug%03d.png" % counter, image)

    # Log
    logger.debug(u"Debug output written to debug%03d.png", counter)

    # Return counter + 1
    return counter + 1


# Function: Sorts coordinates by distance to a given point
def sortCoordinatesByDistanceToPoint(coordinates, (x, y)):
    """ Sorts a given list of `coordinates` by distance to the point `(x,y)`. 'coordinates' should be a numpy array.
    Returns a sorted list (numpy array) or an empty list if an error occurred.
    """
    # No numpy array?
    if type(coordinates) is not np.ndarray:
        logger.error(u"sortCoordinatesByDistanceToPoint(coordinates, (x, y)): coordinates should be a "
                     u"numpy array of coordinates")
        return []

    # Convert to list
    sortedlist = coordinates.tolist()

    # Sort the coordinates by distance to (x, y)
    sortedlist.sort(key=lambda item: math.hypot(item[0] - x, item[1] - y))

    # Return (converting to numpy.array again)
    return np.array(sortedlist)


# Function: Sort the coordinates, so that the order will be top-left, top-right, bottom-left, bottom-right
def sortCoordinates(box, (width, height)):
    """ Sorts the coordinates of a given rectangle (`box` given as numpy array), so that the order will be top-left,
    top-right, bottom-left, bottom-right. `width` and `height` of image or surrounding rectangle. Returns the sorted
    coordinates or an empty list if an error occurred.
    """
    # No integer?
    if type(width) is not int or type(height) is not int:
        logger.error(u"sortCoordinates(box, (width, height)): width and height should be integers")
        return []

    # No numpy array?
    if type(box) is not np.ndarray:
        logger.error(u"sortCoordinates(box, (width, height)): box should be a numpy array of four coordinates.")
        return []

    # Create a sorted box
    sortedbox = []

    # Find top-left corner: It is the nearest point to (0, 0)
    sortedbox.append(sortCoordinatesByDistanceToPoint(box, (0, 0))[0])

    # Find top-right corner
    sortedbox.append(sortCoordinatesByDistanceToPoint(box, (width, 0))[0])

    # Find bottom-left corner
    sortedbox.append(sortCoordinatesByDistanceToPoint(box, (0, height))[0])

    # Find bottom-right corner
    sortedbox.append(sortCoordinatesByDistanceToPoint(box, (width, height))[0])

    # Return sorted box
    return sortedbox


# Function: Tries to find the flow markers on a given image; gives a tuple with indices of the contours
def findFlowMarkers(contours, (outerzoneindex, innerzoneindex), zonebox, areavariance):
    """ Tries to find the flow markers on a given image. `contours` is the respective return value of
    `cv2.findContours`. `outerzoneindex` and `innerzoneindex` are the indices of the outer and inner rectangle of the
    separation zone (found by `ffe.findSeparationZoneBoundaries`). `zonebox` is the rectangle box of the outer
    separation zone. `areavariance` is the variance the areas of both flowmarkers are allowed to have.

    Returns the indices of both flowmarkers `(int, int)` in `contours` or `-1, -1` if an error occurred.
    """
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

    # Test each contour
    for i in xrange(len(contours)):
        # Exclude parent and childbox
        if i in [outerzoneindex, innerzoneindex]:
            continue

        # Inside parentbox?
        if cv2.pointPolygonTest(np.array([zonebox]), centers[i], False) == +1:
            continue

        # Calculate mirrored point, i.e. (mx, my) = 2*(zx, zy) - (x, y)
        mirrorpt = tuple(np.subtract(np.multiply(2, centers[innerzoneindex]), centers[i]))

        # Mirrored point inside parentbox?
        if cv2.pointPolygonTest(np.array([zonebox]), mirrorpt, False) == +1:
            continue

        # Check all contours after i
        for j in xrange(len(contours) - i - 1):
            # Test if point is inside of polygon (contour)
            if cv2.pointPolygonTest(contours[i + j + 1], mirrorpt, False) == +1:
                area1 = float(cv2.contourArea(contours[i]))
                area2 = float(cv2.contourArea(contours[i + j + 1]))
                if area1 == 0.0 or area2 == 0.0:
                    continue

                # Maybe candidate
                logger.info(u"Mirrored center point of contour %d (area %.2f) is inside contour %d (area %.2f).",
                            i, area1, j + i + 1, area2)

                # Calculate are fraction of area difference
                areadiff = (max(area1, area2) - min(area1, area2)) / max(area1, area2)

                # Test if the fraction is in the allowed range
                if areadiff <= areavariance:
                    logger.info(u"Area difference is in the allowed range (%.2f <= %.2f)", areadiff, areavariance)
                    return (i, j + i + 1)

    # Nothing found?
    return (-1, -1)


# Function: Extracts a (separation) zone from an image
def extractZoneFromImage(image, zone, border):
    """Extracts a `zone` such as the separation zone from an `image` and returns it with a blackened border (to remove
    the markings). `zone` is a list with four coordinates as tuples for the boundary in the order of top-left,
    top-right, bottom-left, bottom-right. `border` is the number of pixels to blacken.

    The extracted zone is rotated (warp-transformed) if needed. Returns the image (numpy array) as well as the
    transformation matrix or simply `None` if an error occurred.
    """
    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"extractZoneFromImage(image, zone, border): image is not a numpy array")
        return None

    # Is zone a list of 4 coordinates?
    if type(zone) is not list or len(zone) != 4:
        logger.error(u"extractZoneFromImage(image, zone, border): zone is not a list of 4 tuples")
        return None

    # Is border an int?
    if type(border) is not int or border < 0:
        logger.error(u"extractZoneFromImage(image, zone, border): border is not a positive integer or 0")

    # Width is the distance between the top-left (index: 0) and top-right (index: 1)
    # OR the distance between bottom-left (index: 2) and bottom-right (index: 3)
    # The maximum will be selected.
    width = int(max(math.hypot(zone[1][0]-zone[0][0], zone[1][1]-zone[0][1]),
                    math.hypot(zone[3][0] - zone[2][0], zone[3][1] - zone[2][1])))

    # Height is the distance between the top-left and bottom-left
    # OR the distance between top-right and bottom-right
    # Again, the maximum will be selected.
    height = int(max(math.hypot(zone[2][0] - zone[0][0], zone[2][1] - zone[0][1]),
                     math.hypot(zone[3][0] - zone[1][0], zone[3][1] - zone[1][1])))

    # Source and target points for the perspective transform
    sourcepoints = np.float32(zone)
    targetpoints = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Create matrix for transformation
    transmatrix = cv2.getPerspectiveTransform(sourcepoints, targetpoints)

    # Perform transformation
    zoneimage = cv2.warpPerspective(image, transmatrix, (width, height))

    # Blacken the border a little bit
    cv2.rectangle(zoneimage, (0, 0), (width, height), (0, 0, 0), border)

    return zoneimage, transmatrix


# Function: Calculates the inverse flow direction based on flow markers
def getInverseFlowDirection(markers, transmatrix):
    """Calculates the inverse flow direction for the extracted zone based on the flow markers and the
    transformation matrix from `ffe.extractZoneFromImage`. Returns the normalized, inverse flow direction as tuple of
    floats or `(-1.00, 0.00)`.
    """
    # Is transmatrix an numpy array?
    if type(transmatrix) is not np.ndarray:
        logger.error(u"getInverseFlowDirection(markers, transmatrix): transmatrix is not an numpy array")
        return None

    # Is zone a list of 4 coordinates?
    if type(markers) is not list or len(markers) != 2:
        logger.error(u"getInverseFlowDirection(markers, transmatrix): markers is not a list")
        return None

    # Flow direction
    inverseflow = (-1.00, 0.00)

    # Extend the dimension of the 2D markers to a third dimension (the transformation matrix is 3x3!)
    markers[0] = np.append(np.array(markers[0]), [1])
    markers[1] = np.append(np.array(markers[1]), [1])

    # Transform markers (dot-product of matrix and vector)
    markers[0] = transmatrix.dot(markers[0])
    markers[1] = transmatrix.dot(markers[1])

    # Calculate the difference vector
    vector = np.array([markers[0][0]-markers[1][0], markers[0][1]-markers[1][1]])

    # Get normalization for this vector
    norm = np.linalg.norm(vector)

    # Normalize if possible and return
    if norm != 0:
        vector /= norm
        inverseflow = tuple(vector)

    # Return inverse flow
    return inverseflow


# Function: Finds the middle points and width of consecutive number groups in a list
def findMiddleOfConsecutiveNumbers(numbers):
    """Finds the centre and width of consecutive numbers in a list. Let's assume we have a list of integers such as

    `[1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6]`.

    Then, this function will return a list of tuples in the form of (index of centre, width - both as integers!):

    `[(1, 1), (6, 3), (10, 0), (12, 1)]`.
    """
    # Is numbers a list?
    if type(numbers) is not list:
        logger.error(u"findMiddleOfConsecutiveNumbers(numbers): numbers is not a list")
        return []

    # Create empty list of results
    results = []

    # With the help of itertools, find the ranges; This is a receipe from the itertools-docs:
    for k, g in itertools.groupby(enumerate(numbers), lambda (i, x): i - x):
        # Get group
        group = map(operator.itemgetter(1), g)
        # Middle index
        mindex = int(len(group) / 2)
        # Add the "middle" point of the group and the total width of the group to the results
        results.append((group[mindex], len(group)))

    # Return the results
    return results


# Function: Tries to find the endpoints of trajectories
def findEndpointsOfTrajectories(image, threshpenetration=0.70, ordermin=0.05):
    """Finds the endpoints of trajectories by light casting. The input is an 8bit `image` (numpy array).
    `threshpenetration` sets the maximum travel of the light from right to left on the image (in percentage,
    default: 0.70). `ordermin` is the minimum number of points (relative to width of image) a minimum must possess to
    be considered as minimum.

    Returns a list of endpoints (as tuple in the format of `(x, y, width)`) or an empty list if an error occurred.
    """
    global debugmode

    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"findEndpointsOfTrajectories(image, threshpenetration): image is not a numpy array")
        return []

    # Is threshpenetration a float?
    if type(threshpenetration) is not float:
        logger.error(u"findEndpointsOfTrajectories(image, threshpenetration): threshpenetration is not a float")
        return []

    # Get dimensions of image
    height, width = image.shape

    # Empty list
    endpoints = []

    # This is a list of "nonzero" coordinates (transposed, so we get an ordered array)
    nz = np.transpose(image.nonzero())

    # Create an array with the dimensions (height of image, 1), and fill it with (width of image)
    lines = np.full((height, 1), width, dtype=np.int32)

    # Now overwrite the lines with the values from the nz-array. Since is ordered, we do not have to care
    # about filtering - in the end, the max. x coordinate of each line is saved
    for c in nz:
        lines[c[0]] = (width - c[1]) if c[1] >= (width - int(threshpenetration * width)) else width

    # Try to find local minimas; critical parameter: order (how many points to compare)
    indices_raw = scipy.signal.argrelextrema(lines, np.less_equal, order=max(int(ordermin*height), 1))[0]

    # Remove all indices, for which the corresponding point is out of limit (threshpenetration)
    indices = []
    for index in indices_raw:
        if lines[index] <= int(threshpenetration*width):
            indices.append(index)

    # Now, find out the consecutive number groups in this list
    groups = findMiddleOfConsecutiveNumbers(indices)

    # Now compile this into endpoints; recalculates the values (be aware: x and y are exchanged here!)
    for g in groups:
        # Also, we use half of the width as 1st-order-approximation for the FWHM
        endpoints.append((int(width-lines[g[0]]-1), g[0], max(int(g[1]/2), 1)))

    # Debug?
    if debugmode:
        # Create new image, put 8bit image in green
        debugimage = np.zeros((height, width, 3), np.uint8)
        debugimage[:, :, 1] = image

        # Draw lines on image
        for l in xrange(len(lines)):
            cv2.line(debugimage, (width-1, l), (width - lines[l] - 1, l), (255, 215, 0))

        # Plot (be aware that this pauses the program)
        plt.figure(num=None, figsize=(2 * 4, 1 * 4), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(1, 2, 1)
        plt.xlabel('x [Pixel]')
        plt.ylabel('y [Pixel]')
        plt.imshow(debugimage)

        plt.subplot(1, 2, 2)
        plt.plot(lines)
        for g in groups:
            plt.plot([g[0]], [lines[g[0]]], 'ro')
        plt.ylabel("x-penetration (from right)")
        plt.xlabel("y (from top to bottom)")
        plt.axis([0, height, 0, width])

        plt.show()

    # Return the list of points
    return endpoints


# Function: Find trajectories on 8bit image, given the start and endpoints, by backtracking
def getTrajectoriesFromImage(image, startpoints, endpoints, flow, maxiteration=1000, gradientfactor=0.10,
                             densityrad=1, maxattraction=1.0):
    """Finds trajectories on a 8bit `image` (numpy array) using a gradient method. `startpoints` is a list of
    coordinates for the inlets. `endpoints` are the endpoints of the trajectories detected by
    `ffe.findEndpointsOfTrajectories` or simply the outlets. `flow` is the starting vector of the slope (usually
    the inverse hydrodynamic flow vector). `maxiteration` defines the maximum iterations (default: 1000).
    `gradientfactor` defines the step size of the algorithm as a scaling factor (default: 0.10), the gradient itself
    (the step size in pixels) is calculated by

    `gradient = max(int(gradientfactor * max(width, height)), 2)`

    `densityrad` is the radius of the density field to calculate; 1 = 3x3 field, 2 = 5x5 field, etc (default: 1).
    `maxattraction` is a factor to control the attraction of the inlets (default: 1.0).

     Returns a list with the coordinates and a standard width (10) of the trajectories in the form of `(x, y, w)`.
    """
    global debugmode

    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"getTrajectoriesFromImage(image, ...): image is not a numpy array")
        return []

    # Is startpoints a list?
    if type(startpoints) is not list:
        logger.error(u"getTrajectoriesFromImage(..., startpoints, ...): startpoints is not a list")
        return []

    # Is endpoints a list?
    if type(endpoints) is not list:
        logger.error(u"getTrajectoriesFromImage(..., endpoints, ...): endpoints is not a list")
        return []

    # Is flow a tuple?
    if type(flow) is not tuple:
        logger.error(u"getTrajectoriesFromImage(..., flow, ...): flow is not a tuple")
        return []

    # Is maxiteration a int?
    if type(maxiteration) is not int:
        logger.error(u"getTrajectoriesFromImage(..., maxiteration, ...): maxiteration is not a int")
        return []

    # Is gradientfactor a float?
    if type(gradientfactor) is not float:
        logger.error(u"getTrajectoriesFromImage(..., gradientfactor, ...): gradientfactor is not a float")
        return []

    # Is densityrad a int?
    if type(densityrad) is not int:
        logger.error(u"getTrajectoriesFromImage(..., densityrad, ...): densityrad is not a int")
        return []

    # Is maxattraction a float?
    if type(maxattraction) is not float:
        logger.error(u"getTrajectoriesFromImage(..., maxattraction, ...): maxattraction is not a float")
        return []

    # Image shape
    height, width = image.shape

    # Create empty array of trajectories
    trajectories = []

    # Start clock
    starttime = time.clock()

    # For each endpoint, we start a new search
    for endpoint in endpoints:
        # Create empty array for the trajectory
        trajectory = []

        # Running?
        run = True
        i = 0

        # Starting slope is the direction of the flow
        slope = tuple(flow)

        # First point is the actual endpoint
        trajectory.append(endpoint)

        # Calculate the initial gradient; at least 2
        gradient = max(int(gradientfactor * max(width, height)), 2)

        # Start the algorithm!
        while run:
            # The current point is the last point we added; bring it in focus
            targetpoint = (trajectory[-1][0] + int(slope[0]*gradient), trajectory[-1][1] + int(slope[1]*gradient))

            # Now calculate the top-left point of a rectangle around this point, with gradient as edge
            topleftpoint = (targetpoint[0] - int(gradient/2), targetpoint[1] - int(gradient/2))

            # Create a slice of the image to work with (the plus is important: create a new reference!)
            imageslice = copy.copy(image[topleftpoint[1]:(topleftpoint[1]+gradient),
                                   topleftpoint[0]:(topleftpoint[0]+gradient)])

            # Image slice does not contain any values? Break!
            if imageslice.size == 0:
                break

            # Empty? Break!
            if imageslice.max() == 0:
                break

            # It might be that the target point lies in nirvana -> move to nearest not-null point
            if imageslice[int(gradient/2), int(gradient/2)] == 0:
                # Find all non-zero values
                nzvalues = np.transpose(imageslice.nonzero())

                # Order them by distancce to the current point
                nzvalues = sortCoordinatesByDistanceToPoint(nzvalues, (int(gradient / 2), int(gradient / 2)))

                # Move the current point AND focus
                targetpoint = (topleftpoint[0] + nzvalues[0][1], topleftpoint[1] + nzvalues[0][0])

                # Now recalculate the top-left point of a rectangle around this point, with gradient as edge
                topleftpoint = (targetpoint[0] - int(gradient / 2), targetpoint[1] - int(gradient / 2))

                # Create a new slice of the image to work with
                imageslice = copy.copy(image[topleftpoint[1]:(topleftpoint[1] + gradient),
                                       topleftpoint[0]:(topleftpoint[0] + gradient)])

            # Create an empty image for density calculations
            densityimage = np.zeros((gradient - 2 * densityrad, gradient - 2 * densityrad), dtype=np.int32)

            # Calculate the density
            for x in xrange(gradient-2*densityrad):
                for y in xrange(gradient-2*densityrad):
                    # The point in the middle should be not zero (we do not want to create points!)
                    if not imageslice[y, x] == 0:
                        densityimage[y, x] = imageslice[y-densityrad:y+densityrad, x-densityrad:x+densityrad].sum()
                        densityimage[y, x] /= (2*densityrad + 1)**2

            # Find the maximum(s) of density
            maxvalues = np.transpose(np.where(densityimage == densityimage.max()))

            # Sort this list by distance from the middle point of this slice
            maxvalues = sortCoordinatesByDistanceToPoint(maxvalues, (int(gradient / 2), int(gradient / 2)))

            # Calculate target point
            targetpoint = (topleftpoint[0] + densityrad + maxvalues[0][1],
                           topleftpoint[1] + densityrad + maxvalues[0][0])

            # Calculate new slope
            slope = (targetpoint[0]-trajectory[-1][0], targetpoint[1]-trajectory[-1][1])
            if np.linalg.norm(slope) != 0:
                slope /= np.linalg.norm(slope)

            # The slope is modified by an virtual "attraction force" of the inlets; the nearer the point to
            # the inlet, the stronger the force
            for inlet in startpoints:
                # Decay radius
                d_half = ((inlet[2]+gradient)/2.0)/0.1

                # Distance from targetpoint to inlet
                distance = math.hypot(targetpoint[0]-inlet[0], targetpoint[1]-inlet[1])

                # Calculate attraction vector
                attrv = (inlet[0]-targetpoint[0], inlet[1]-targetpoint[1])
                if np.linalg.norm(attrv) != 0:
                    attrv /= np.linalg.norm(attrv)

                # Calculate the intensity of attraction
                attraction = maxattraction*math.exp(-distance/d_half)

                # Modify slope
                slope = (slope[0]+attraction*attrv[0], slope[1]+attraction*attrv[1])
                if np.linalg.norm(slope) != 0:
                    slope /= np.linalg.norm(slope)

            # Add to our trajectory
            trajectory.append((targetpoint[0], targetpoint[1], 10))

            # Test if targetpoint is in one of the startpoints (x, y, diameter), if yes, break
            for pt in startpoints:
                dist = math.hypot(targetpoint[0]-pt[0], targetpoint[1]-pt[1])
                if dist <= ((pt[2]/2.0)+gradient):
                    # Add startpoint
                    trajectory.append(pt)
                    # Turn off
                    run = False

            # Iteration
            i += 1
            if i > maxiteration:
                run = False

        # Add trajectory, if not empty and at least two points!
        if len(trajectory) > 1:
            # Append inverse of list (i.e. the trajectory starts left (inlet) and goes to the right (outlet/border))
            trajectories.append(trajectory[::-1])

    endtime = time.clock()

    if debugmode:
        logger.info(u"Time needed: %.3f seconds" % (endtime - starttime))

    # Return trajectories
    return trajectories


# Function: Calculates a bonding rectangle of a trajectory (not the minimum rectangle!)
def calculateTrajectoryRectangle(trajectory):
    """Calculates a bonding rectangle of a `trajectory` (not the minimum!). `trajectory` should be a list of
    coordinates with widths in the form of `(x, y, w)`-tuples.

    Returns a list of two tuples width the coordinates of the top-left and the bottom-right point of the rectangle or
    an empty list if an error occurred.
    """
    # Is tracjectory a list?
    if type(trajectory) is not list:
        logger.error(u"calculateTrajectoryRectangle(trajectory): trajectory has to be a list.")
        return []

    # A trajectory is a list of triples (x, y, width), so it has to be converted into a list of points
    pts = []

    # Add the points; add both x - width, x + width and y - width, and y + width, i.e. the bounding rectangle of
    # the circle with midpoint x,y and diameter width
    for triple in trajectory:
        pts.append([triple[0] - int(triple[2]/2)-1, triple[1] - int(triple[2]/2)-1])
        pts.append([triple[0] + int(triple[2]/2)+1, triple[1] + int(triple[2]/2)+1])

    # Get the bonding rectangle
    x, y, w, h = cv2.boundingRect(np.array(pts))

    # Return the two points
    return [(x, y), (x+w, y+h)]


# Function: Calculates the outer points (i.e. skin) of a trajectory incorporating widths and slopes of the points
def getSkinOfTrajectory(trajectory):
    """Calculates the skin of a `trajectory`. `trajectory` should be a list of coordinates with widths in the
    form of `(x, y, w)`-tuples.

    Returns a list of tuples `(x, y)` width the skin coordinates. The first half of the list are the points on one side
    of the trajectory, the other half the other points (in inverse direction). With this, rendering the full polygon
    is straightforward. The list is empty if an error occurred.
    """
    # Is tracjectory a list?
    if type(trajectory) is not list:
        logger.error(u"getSkinOfTrajectory(trajectory): trajectory has to be a list.")
        return []

    # Trajectory needs at least two points
    if len(trajectory) < 2:
        logger.error(u"getSkinOfTrajectory(trajectory): trajectory needs at least two points.")
        return []

    # Two lists, for both sides of the skin (arbitrarily named left and right)
    skinleft = []
    skinright = []

    # Last slope = slope of first line
    lastslope = (trajectory[1][0]-trajectory[0][0], trajectory[1][1]-trajectory[0][1])
    if np.linalg.norm(lastslope) != 0:
        lastslope /= np.linalg.norm(lastslope)

    # For every point in the trajectory-list, calculate the two skinpoints
    for i in xrange(len(trajectory)):
        # Set up slope (use lastslope as initial value)
        slope = lastslope

        # Calculate slope
        # Not last index? Take next slope
        if i < (len(trajectory)-1):
            slope = (trajectory[i+1][0]-trajectory[i][0], trajectory[i+1][1]-trajectory[i][1])
            if np.linalg.norm(slope) != 0:
                slope /= np.linalg.norm(slope)

        # Now combine the slopes to a skin-vector
        skinvector = (slope[0] - lastslope[0], slope[1] - lastslope[1])
        if np.linalg.norm(skinvector) != 0:
            skinvector /= np.linalg.norm(skinvector)

        # 1. special case: slope and lastslope are identical (first and last index)
        # 2. special case: slope and lastslope (and thus, the skinvector) are (anti)parallel
        # Both cases can be caught by looking at the cross product of slope and lastslope (will be zero)
        # If so, then use just 90°-rotated lastslope
        if np.cross(np.array(slope), np.array(lastslope)) == 0:
            skinvector = (np.array([[0, -1], [1, 0]]).dot(np.array(lastslope)))

        # Calculate two points by adding and subtracting the skinvector, respectively
        pts = [(int(trajectory[i][0] + skinvector[0] * trajectory[i][2] / 2.0),
                int(trajectory[i][1] + skinvector[1] * trajectory[i][2] / 2.0)),
               (int(trajectory[i][0] - skinvector[0] * trajectory[i][2] / 2.0),
                int(trajectory[i][1] - skinvector[1] * trajectory[i][2] / 2.0))]

        # Check if the skinvector is on the left of slope-vector (by cross-product)
        if np.cross(np.array(slope), np.array(skinvector)) > 0:
            skinleft.append(pts[0])
            skinright.append(pts[1])

        # Otherwise, do it the other way around
        else:
            skinleft.append(pts[1])
            skinright.append(pts[0])

        # Set lastslope
        lastslope = slope

    # In the end,  combine the fist list and the inverse of the second one -> full polygon
    return skinleft + skinright[::-1]


# Function: Calculates the overlap between two trajectories
def calculateTrajectoriesOverlap(trac1, trac2):
    """Calculates the overlap between two trajectories `trac1` and `trac2`. For this, the function will
    render both trajectories on a black image using green and red color, respectively. The overlapping region will
    be yellow. The overlap can then be calculated simply by:

    `overlap = yellow/min(green, red)`.

    Returns a `float` between `0.0` and `1.0`.
    """
    # Is trac1 and trac2 a list?
    if type(trac1) is not list or type(trac2) is not list:
        logger.error(u"calculateTrajectoriesOverlap(trac1, trac2): trac1 and trac2 have to be lists.")
        return 0.0

    # Empty?
    if len(trac1) == 0 or len(trac2) == 0:
        logger.warning(u"calculateTrajectoriesOverlap(trac1, trac2): empty trajectories given.")
        return 0.0

    # The idea here is to render both trajectories with a basic rendering method to a RGB image; one trajectory
    # is blue, the other one is red; the overlap is yellow - in the end, just have to count pixels!

    # First, calculate the bonding rect neeeded to draw both trajectories
    rect = calculateTrajectoryRectangle(trac1 + trac2)

    # Calculate the width and height of this rectangle
    width, height = rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]

    # Create twp empty images, 24bit
    image1 = np.zeros((height, width, 3), np.uint8)
    image2 = np.zeros((height, width, 3), np.uint8)

    # Get the skins of the trajectories
    skin1 = np.array(getSkinOfTrajectory(trac1))
    skin2 = np.array(getSkinOfTrajectory(trac2))

    # No skin?
    if len(skin1) == 0 or len(skin2) == 0:
        logger.warning(u"calculateTrajectoriesOverlap(trac1, trac2): Could not calculate skin.")
        return 0.0

    # Offset is the top-left corner of the rectangle
    skin1 = np.subtract(skin1, np.array(rect[0]))
    skin2 = np.subtract(skin2, np.array(rect[0]))

    # Draw the filled polygons, one in green, and one in red
    cv2.fillPoly(image1, [skin1], (0, 255, 0))
    cv2.fillPoly(image2, [skin2], (0, 0, 255))

    # Count green and red pixels in the corresponding images and sum up
    green = cv2.countNonZero(cv2.inRange(image1, (0, 255, 0), (0, 255, 0)))
    red = cv2.countNonZero(cv2.inRange(image2, (0, 0, 255), (0, 0, 255)))

    # No pixels?
    if min(green, red) == 0:
        logger.warning(u"calculateTrajectoriesOverlap(trac1, trac2): No trajectories were drawn; can't compare areas.")
        return 0.0

    # Add both images
    image1 = cv2.add(image1, image2)

    # Count yellow pixels
    yellow = cv2.countNonZero(cv2.inRange(image1, (0, 255, 255), (0, 255, 255)))

    if debugmode:
        logger.debug("Overlap: %s, %s and %s, %s; %d, %d, %d, %.2f", str(trac1[0]), str(trac1[-1]),
                     str(trac2[0]), str(trac2[-1]),
                     green, red, yellow, float(yellow)/float(min(green, red)))

    # No yellow pixels means no overlap
    if yellow == 0:
        return 0.0

    if debugmode:
        cv2.imshow('Overlap', image1)
        cv2.waitKey(5000)

    # The overlap is basically yellow pixels over total pixels.
    # However, there might be the situation that one trajectory is smaller and still (completely) inside of
    # the other trajectory. Then, this simple approach does not work. So, it is better to calculate the overlap
    # by yellow pixels over smallest trajectory
    return float(yellow)/float(min(green, red))



# Function: Calculates the distance between two trajectories
def calculateTrajectoriesDistanceHausdorff(trac1, trac2):
    """Calculates the Hausdorff distance between two trajectories `trac1` and `trac2`. This is an alternative to
    `ffe.calculateTrajectoriesOverlap` to check if two polygons (in this case trajectories) are similar or equal.
    Similar polygons should have a similar Hausdorff distance (Please see
    [Wikipedia](https://en.wikipedia.org/wiki/Hausdorff_distance) for more information).

    Returns the distance as `float`.
    """
    # Is trac1 and trac2 a list?
    if type(trac1) is not list or type(trac2) is not list:
        logger.error(u"calculateTrajectoriesOverlap(trac1, trac2): trac1 and trac2 have to be lists.")
        return 0.0

    # Empty?
    if len(trac1) == 0 or len(trac2) == 0:
        logger.warning(u"calculateTrajectoriesOverlap(trac1, trac2): empty trajectories given.")
        return 0.0

    # First, convert the trajectory-lists into point lists
    pts1 = np.array([[x, y] for (x, y, d) in trac1])
    pts2 = np.array([[x, y] for (x, y, d) in trac2])

    # Calculate distance matrix
    distance = scipy.spatial.distance.cdist(pts1, pts2, 'euclidean')

    # Calculate Hausdorff distance
    hausdorff = max(float(np.max(np.min(distance, axis=1))), float(np.max(np.min(distance, axis=0))))

    if debugmode:
        logger.info("Hausdorff: %s, %s and %s, %s; %d, %.2f", str(trac1[0]), str(trac1[-1]), str(trac2[0]),
                    str(trac2[-1]), len(distance), hausdorff)

        # First, calculate the bonding rect neeeded to draw both trajectories
        rect = calculateTrajectoryRectangle(trac1 + trac2)

        # Calculate the width and height of this rectangle
        width, height = rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]

        # Create twp empty images, 24bit
        image1 = np.zeros((height, width, 3), np.uint8)
        image2 = np.zeros((height, width, 3), np.uint8)

        # Get the skins of the trajectories
        skin1 = np.array(getSkinOfTrajectory(trac1))
        skin2 = np.array(getSkinOfTrajectory(trac2))

        # Offset is the top-left corner of the rectangle
        skin1 = np.subtract(skin1, np.array(rect[0]))
        skin2 = np.subtract(skin2, np.array(rect[0]))

        # Draw the filled polygons, one in green, and one in red
        cv2.fillPoly(image1, [skin1], (0, 255, 0))
        cv2.fillPoly(image2, [skin2], (0, 0, 255))

        # Add both images
        image1 = cv2.add(image1, image2)

        cv2.imshow('Overlap', image1)
        cv2.waitKey(5000)

    # Calculate and return the Hausdorff-distance based on D
    return hausdorff


# Function: Combines two lists of trajectories; checks for identical or similar trajectories
def combineTrajectories(traclist1, traclist2, maxoverlap=0.75, useHausdorff=False, hausdorffbias=10.0):
    """Combines two lists of trajectories 'traclist1' and 'traclist2' by checking for identical/similar ones. Similar
     ones are only return once. Either overlap calculations (by `ffe.calculateTrajectoriesOverlap`) or Hausdorff
     distance (by `ffe.calculateTrajectoriesDistanceHausdorff`) can be used to compare trajectories.

     `maxoverlap` defines the maximum overlap (default: 0.75) two trajectories can have before they are considered
     identical. `useHausdorff' (default: `False`) and 'hausdorffbias' (default: `10.0`) can be used to use
     Hausendorff distance instead of overlap for comparison.

     Returns a new list containing the combined members of `traclist1` and `traclist2`.
    """
    # Is traclist1 and traclist2 a list?
    if type(traclist1) is not list or type(traclist2) is not list:
        logger.error(u"combineTrajectories(traclist1, traclist2, maxoverlap): traclist1 and "
                     u"traclist2 have to be lists.")
        return []

    # Is maxoverlap a float?
    if type(maxoverlap) is not float:
        logger.error(u"combineTrajectories(traclist1, traclist2, maxoverlap): maxoverlap has to be a float.")
        return []

    # Our working list is the combination of both lists
    workinglist = traclist1 + traclist2

    # Final list is empty at beginning
    finallist = []

    # As long as there is something in the working list do...
    while len(workinglist) > 0:
        # Get and remove an item from this list
        item = workinglist.pop()

        # Present in finallist?
        present = False

        # Check if this item or a similar one is already in the finallist
        for finalitem in finallist:
            # Check criteria for overlap
            if useHausdorff:
                criteria = (calculateTrajectoriesDistanceHausdorff(item, finalitem) <= hausdorffbias)
            else:
                criteria = (calculateTrajectoriesOverlap(item, finalitem) >= maxoverlap)

            # Found something?
            if criteria:
                # Ok, there is a similar trajectory in the finallist already
                present = True

                # Only, when this new trajectory from the working list is larger/longer then the one in the finallist
                # we have the exchange them (overwrite)
                if len(item) > len(finalitem):
                    finallist[finallist.index(finalitem)] = item

                # In any case, break here
                break

        # Only if the item (or sth similar) is not present already, we add it
        if not present:
            finallist.append(item)

    # Return the final list
    return finallist


# Function: Finds a path between two points on a 8bit image
def findPathOnImage(image, startpoint, endpoint, bias=50, everyotherpoint=10, densityrad=4):
    """Finds a path between `startpoint` and `endpoint` on a 8bit `image` using Dijkstra's algorithm. `bias` is the
    minimum intensity density a pixel must have to be considered as passable (otherwise its cost will be set to
    infinite). `densityrad` is the radius of the density field to calculate; 1 = 3x3 field, 2 = 5x5 field, etc
    (default: 1).

    Although the algorithm can return every point in a path, this is not useful to get a smooth path. Therefore, it
    will only return every x. point, where x is given by `everyotherpoint` (default: 10).

    Returns a list with the coordinates of the path (without widths) or an empty list if an error occurred.
    """
    global debugmode

    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"findPathOnImage(image, ...): image is not a numpy array")
        return []

    # Is startpoints a list?
    if type(startpoint) is not tuple:
        logger.error(u"findPathOnImage(..., startpoint, ...): startpoints is not a tuple")
        return []

    # Is endpoints a list?
    if type(endpoint) is not tuple:
        logger.error(u"findPathOnImage(..., endpoint, ...): endpoints is not a tuple")
        return []

    # Get image shape
    height, width = image.shape

    # It is hard to tell a computer to set something to "infinite", so the definition of infinite here is
    # a little bit more real... a number, which the path cannot (or better: should not) reach.
    infinite = int(math.hypot(height, width)*100e6)

    # Create an empty image for density calculations
    densityimage = np.zeros((height, width), dtype=np.int32)

    # Calculate the density
    for x in xrange(width - 2 * densityrad):
        for y in xrange(height - 2 * densityrad):
            # The point in the middle should be not zero (we do not want to create points!)
            if not image[y + densityrad, x + densityrad] == 0:
                densityimage[y + densityrad, x + densityrad] = image[y:y + 2*densityrad, x:x + 2*densityrad].sum()

    # Get maximum of this density-image (for cost calculations)
    maxintensity = densityimage.max()

    # Debug?
    if debugmode:
        logger.info(u"Start point (%d, %d) intensity: %d", startpoint[0], startpoint[1],
                    densityimage[startpoint[1], startpoint[0]])
        logger.info(u"End point (%d, %d) intensity: %d", endpoint[0], endpoint[1],
                    densityimage[endpoint[1], endpoint[0]])

        # Rescale for showing
        showdensity = rescaleFrameTo8bit(densityimage)
        # Convert to 24bit color
        showdensity = cv2.cvtColor(showdensity, cv2.COLOR_GRAY2BGR)
        cv2.circle(showdensity, (startpoint[0], startpoint[1]), 5, (0, 215, 255), -5)
        cv2.circle(showdensity, (endpoint[0], endpoint[1]), 5, (0, 215, 255), -5)
        cv2.imshow("Density", showdensity)
        cv2.waitKey(1)

    # Is the start or end point empty?
    if densityimage[startpoint[1], startpoint[0]] == 0 or densityimage[endpoint[1], endpoint[0]] == 0:
        logger.info(u"Intensity of start or end point is zero.")

        # Wait in debug mode
        if debugmode:
            cv2.waitKey(1000)

        return []

    # Get a list of all points above the bias (excluding!)
    points = np.transpose(np.where(densityimage > bias)).tolist()

    # Create a list with tuples: (x, y, distance)
    # Set all distances to infinite except for the startpoint
    queue = []
    for pt in points:
        distance = infinite
        if pt[1] == startpoint[0] and pt[0] == startpoint[1]:
            distance = 0
        heapq.heappush(queue, (distance, pt[1], pt[0]))

    # Create an empty "image", which holds the indices for backtracking
    tracker = np.zeros((height, width), dtype=np.int32)

    # Give the end point a negative index
    tracker[endpoint[1], endpoint[0]] = -1

    # As long this queue is not empty, do...
    while len(queue) > 0:
        # Pop (get and remove) item with smallest distance in queue
        firstitem = heapq.heappop(queue)

        # Endpoint reached?
        if firstitem[1] == endpoint[0] and firstitem[2] == endpoint[1]:
            break

        # Find all neighbours of firstitem in queue
        neighbours = [node for node in queue if int(math.hypot(node[1]-firstitem[1], node[2]-firstitem[2])) == 1]

        needtoheapify = False

        # Update the distance of these neighbours
        for neighbour in neighbours:
            # The higher the intensity of the pixel, the lower the cost
            cost = (maxintensity - densityimage[neighbour[2], neighbour[1]])

            # Calculate a distance: base distance from first item plus its cost
            altdistance = firstitem[0] + cost

            # Shorter? Then update!
            if altdistance < neighbour[0]:
                # Create new tuple
                updneighbour = (altdistance, neighbour[1], neighbour[2])

                # Delete old version
                del queue[queue.index(neighbour)]

                # Deletion of item is INVARIANT to a heap queue, so mark for re-heapifying
                needtoheapify = True

                # Push new version
                heapq.heappush(queue, updneighbour)

                # Update the tracker
                tracker[neighbour[2], neighbour[1]] = points.index(
                    next(item for item in points if item[1] == firstitem[1] and item[0] == firstitem[2]))

        # Need to heapify again? (i.e. item was deleted)
        if needtoheapify:
            heapq.heapify(queue)

    # Give the start point a negative index (just in case)
    tracker[startpoint[1], startpoint[0]] = -1

    # If the the endpoint is still not reached, then there is no path :(
    if tracker[endpoint[1], endpoint[0]] == -1:
        logger.info(u"No path found.")
        return []

    # Now, we just have to backtrack from endpoint to startpoint
    path = [[endpoint[0], endpoint[1]]]

    # Fill the list
    current = next(item for item in points if item[1] == endpoint[0] and item[0] == endpoint[1])

    # Count items
    n = 0

    # Not at start point yet?
    while tracker[current[0], current[1]] != -1 and not (current[1] == startpoint[0] and current[0] == startpoint[1]):
        # From where did we came to this point?
        current = points[tracker[current[0], current[1]]]

        # Add point to list, at the beginning; always add start point (zero distance)
        if (n % everyotherpoint) == 0 or (current[1] == startpoint[0] and current[0] == startpoint[1]):
            path.insert(0, [current[1], current[0]])

        # Increase iteration
        n += 1

    # Debug output
    if debugmode:
        logger.info(u"Dijkstra found %d points for trajectory", len(path))
        cv2.imshow("Tracking", tracker.astype(np.uint8))
        cv2.waitKey(10)

    # If path has only one point? Then dump the "path"
    if len(path) < 2:
        return []

    # Return path
    return path


# Function: Find trajectories on 8bit image, given the start and endpoints, by using pathfinding
def getTrajectoriesFromImageDijkstra(image, startpoints, endpoints, bias=50, everyotherpoint=10, densityrad=4,
                                     spvariancefactor=2):
    """Finds trajectories on a 8bit `image` using Dijkstra's algorithm. It will subsequently check for a path between
    every start and end point given by `startpoints` and `endpoints`.

    Both points should not have zero intensity. Since the endpoint are usually given by
    `ffe.findEndpointsOfTrajectories`, this is usually not the case. However, due to precision loss when converting
    from pixels to milimeter (and vice-versa) and a small diameter inlet, the coordinate of a start point could point
    to an empty pixel. In this case, a point near this coordinate is looked for, which has intensity density above
    `bias` and is not more than `width*spvariancefactor` pixels away. If found, this point will be used instead.

    For an explanation of `bias`, `everyotherpoint`, and `densityrad`, see `ffe.findPathOnImage`.

    Returns a trajectory list of coordinates and a standard width in the form of `(x, y, w)`.
    """
    global debugmode

    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"getTrajectoriesFromImageDijkstra(image, ...): image is not a numpy array")
        return []

    # Is startpoints a list?
    if type(startpoints) is not list:
        logger.error(u"getTrajectoriesFromImageDijkstra(..., startpoints, ...): startpoints is not a list")
        return []

    # Is endpoints a list?
    if type(endpoints) is not list:
        logger.error(u"getTrajectoriesFromImageDijkstra(..., endpoints, ...): endpoints is not a list")
        return []

    # Image is empty?
    if image.max() == 0:
        logger.error(u"getTrajectoriesFromImageDijkstra(image, ...): image is empty.")
        return []

    # Create empty array of trajectories
    trajectories = []

    # For each endpoint, we start a new search
    for endpoint in endpoints:

        # Find the path for each startpoint
        for startpoint in startpoints:
            # First check if start point (usually inlet provided by user) has zero intensity.
            if image[startpoint[1], startpoint[0]] == 0:
                # Get a list of all points above the bias
                points = np.transpose(np.where(image > bias))

                # Switch x and y axis (numpy's first coordinate is y not x)
                points = np.array([[item[1], item[0]] for item in points])

                # Sort this list by distance to startpoint
                points = sortCoordinatesByDistanceToPoint(points, (startpoint[0], startpoint[1]))

                # If the distance of first point to the startpoint is less then (startpointwidth*spvariancefactor),
                # then this will be the new start point!
                if (np.hypot(startpoint[0]-points[0][0], startpoint[1]-points[0][1])) <= \
                        (startpoint[2]*spvariancefactor):
                    startpoint = (points[0][0], points[0][1], startpoint[2])

            starttime = time.clock()
            # Create empty array for the trajectory
            path = findPathOnImage(image, endpoint, startpoint, bias, everyotherpoint, densityrad)
            endtime = time.clock()

            if debugmode:
                logger.info(u"Time needed: %.3f seconds" % (endtime-starttime))

            # Append
            if len(path) > 0:
                # Append reversed path, since we want it to start at the inlet (startpoint) and end at the endpoint
                trajectories.append([(x, y, 10) for [x, y] in path[::-1]])

    # Return the trajectories
    return trajectories


# Function: Determines the widths of a stream trajectory
def findWidthOfTrajectory(image, trajectory, densityrad=1):
    """Extracts the widths of a `trajectory` out of `image`. `densityrad` is the radius of the density field
    to calculate; 1 = 3x3 field, 2 = 5x5 field, etc (default: 1).

    Returns the `trajectory` as list with determined widths or an empty list if an error occurred. Will not modify
    the coordinates.
    """
    # Is image an numpy array?
    if type(image) is not np.ndarray:
        logger.error(u"findWidthOfTrajectory(image, trajectory): image is not a numpy array")
        return []

    # Is trajectory a list?
    if type(trajectory) is not list:
        logger.error(u"findWidthOfTrajectory(image, trajectory): trajectory has to be a list.")
        return []

    # Trajectory needs at least two points
    if len(trajectory) < 2:
        logger.error(u"findWidthOfTrajectory(image, trajectory): trajectory needs at least two points.")
        return []

    # Get dimensions of image
    height, width = image.shape

    # Create an empty image for density calculations
    densityimage = np.zeros((height, width), dtype=np.int32)

    # Calculate the density
    for x in xrange(width - 2 * densityrad):
        for y in xrange(height - 2 * densityrad):
            # The point in the middle should be not zero (we do not want to create points!)
            if not image[y + densityrad, x + densityrad] == 0:
                densityimage[y + densityrad, x + densityrad] = image[y:y + 2 * densityrad, x:x + 2 * densityrad].sum()

    # Maximum width (half height of image probably is never reached)
    maxwidth = int(height / 2)

    # New list for the trajectory
    newtrajectory = []

    # Last slope = slope of first line
    lastslope = (trajectory[1][0] - trajectory[0][0], trajectory[1][1] - trajectory[0][1])
    if np.linalg.norm(lastslope) != 0:
        lastslope /= np.linalg.norm(lastslope)

    # For every point in the trajectory-list, calculate the two skinpoints
    for i in xrange(len(trajectory)):
        # Set up slope (use lastslope as initial value)
        slope = lastslope

        # Calculate slope
        # Not last index? Take next slope
        if i < (len(trajectory) - 1):
            slope = (trajectory[i + 1][0] - trajectory[i][0], trajectory[i + 1][1] - trajectory[i][1])
            if np.linalg.norm(slope) != 0:
                slope /= np.linalg.norm(slope)

        # Now combine the slopes to a skin-vector
        skinvector = (slope[0] - lastslope[0], slope[1] - lastslope[1])
        if np.linalg.norm(skinvector) != 0:
            skinvector /= np.linalg.norm(skinvector)

        # 1. special case: slope and lastslope are identical (first and last index)
        # 2. special case: slope and lastslope (and thus, the skinvector) are (anti)parallel
        # Both cases can be caught by looking at the cross product of slope and lastslope (will be zero)
        # If so, then use just 90°-rotated lastslope
        if np.cross(np.array(slope), np.array(lastslope)) == 0:
            skinvector = (np.array([[0, -1], [1, 0]]).dot(np.array(lastslope)))

        # Calculate two points far away from the target point on the line given by skinvector and maxwidth
        w1 = (trajectory[i][0] + maxwidth * skinvector[0],
              trajectory[i][1] + maxwidth * skinvector[1])
        w2 = (trajectory[i][0] - maxwidth * skinvector[0],
              trajectory[i][1] - maxwidth * skinvector[1])

        # Distance between these two points
        wdistance = int(np.hypot(w1[0] - w2[0], w1[1] - w2[1]))

        # Get all the pixel coordinates along this line
        wpoints = np.transpose([np.linspace(w1[0], w2[0], wdistance).astype(np.int),
                                np.linspace(w1[1], w2[1], wdistance).astype(np.int)])

        # Values along this line
        wvalues = []
        for x, y in wpoints:
            if 0 <= x < width and 0 <= y < height:
                wvalues.append(densityimage[y, x])
            else:
                wvalues.append(0)

        # If wvalues is zero: continue with next point
        if len(wvalues) == 0:
            continue

        # Convert to numpy array
        wvalues = np.array(wvalues)

        # Standard stream width is zero
        streamwidth = 0

        # Mid index (this is were the point itself is) (maxwidth = len(wvalues)/2)
        midindex = maxwidth

        # Divide the data point array into two subarrays, for the left (inverted) and right side
        wvaluesleft, wvaluesright = wvalues[midindex:0:-1], wvalues[midindex:]

        # Find the first indices where the value becomes zero
        indexleft, indexright = np.argwhere(wvaluesleft == 0)[0][0], np.argwhere(wvaluesright == 0)[0][0]

        # Now create a new array with the combined values
        wvalues = np.append(wvaluesleft[indexleft:0:-1], wvaluesright[0:indexright + 1])

        # Update middle index
        midindex = indexleft

        # Try to find local minimas; critical parameter: order (how many points to compare; 3 is a good value here)
        indices_raw = scipy.signal.argrelextrema(wvalues, np.less_equal, order=3)[0]

        # Now, find out the consecutive number groups in this list
        groups = findMiddleOfConsecutiveNumbers(indices_raw.tolist())

        # The width is not interesting, remove it
        indices = [item[0] for item in groups]

        # Find the group for midindex
        for g in xrange(len(indices) - 1):
            # Midindex in this area?
            if indices[g] <= midindex <= indices[g + 1]:
                # Use the higher border value as base line
                base = max(wvalues[indices[g]], wvalues[indices[g + 1]])

                # Change wvalues a little bit, only care about the region of interest
                wvalues = np.array([item - base if indices[g] <= index <= indices[g + 1] and item > base else 0 for index, item
                                    in enumerate(wvalues)])

        # We need at least four values to be able to try to find a spline-approximation with k=3 (cubic)
        if len(wvalues) > 4:
            # We want to find the Full Width At Half Maximum, i.e. the data has to be shifted by the half maximum
            # Then, a spline (cubic, k=3) is used to approximate these data points
            spline = scipy.interpolate.UnivariateSpline(list(range(len(wvalues))), wvalues - wvalues.max()/2, ext=1)

            # Since the Half Height is now on y=0, all we need is to determine the zeros (roots)
            # Ideally, there should only be two; the difference is the width of the stream
            roots = spline.roots()
            if len(roots) > 1:
                for r in xrange(len(roots)-1):
                    # Boundaries are rounded to include border cases
                    if round(roots[r]) <= midindex <= round(roots[r+1]):
                        streamwidth = roots[r+1] - roots[r]
                        break

        if debugmode:
            # Make a copy of the original image
            debugimage = copy.copy(image)

            # Convert to 24bit color
            debugimage = cv2.cvtColor(debugimage, cv2.COLOR_GRAY2BGR)

            # Draw the line and the center point
            cv2.line(debugimage, (int(w1[0]), int(w1[1])), (int(w2[0]), int(w2[1])), (0, 215, 255), 1)
            cv2.circle(debugimage, (trajectory[i][0], trajectory[i][1]), 5, (0, 215, 255), -5)

            # Show image
            cv2.imshow('Slicing', debugimage)

            # Plot values
            plt.plot(wvalues, 'ro-')

            # Plot spline (if possible)
            if len(wvalues) > 4:
                spline = scipy.interpolate.UnivariateSpline(list(range(len(wvalues))),
                                                            wvalues - wvalues.max() / 2, ext=1)

                xs = np.linspace(0, len(wvalues), 1000)
                plt.plot(xs, spline(xs), 'g', lw=2)

                plt.plot([midindex], [wvalues[midindex]], 'og')

                # Find root
                roots = spline.roots()
                if len(roots) > 1:
                    r1, r2 = 0, 1
                    for r in xrange(len(roots) - 1):
                        # Boundaries are rounded to include border cases
                        if round(roots[r]) <= midindex <= round(roots[r + 1]):
                           r1, r2 = roots[r], roots[r + 1]
                           break

                    plt.axvspan(r1, r2, facecolor='g', alpha=0.5)
                    print("roots", spline.roots(), r1, r2, streamwidth)

            # Setup axis and show
            plt.axis([0, len(wvalues), -0.5*densityimage.max(), 1.2*densityimage.max()])
            plt.show()

        # Add the trajectory to new list
        newtrajectory.append((trajectory[i][0], trajectory[i][1], streamwidth))

        # Set lastslope
        lastslope = slope

    # Small attempt to fix zero-width points; works only with a minimum of three points
    if len(newtrajectory) > 2:
        for i in xrange(len(newtrajectory)):
            # Not start or endpoint
            if i == 0 or i == (len(newtrajectory)-1):
                continue

            # Width of the point is zero and of the neighbours are not?
            if newtrajectory[i][2] == 0 and not newtrajectory[i-1][2] == 0 and not newtrajectory[i+1][2] == 0:
                # Average of neighbour widths; update the tuple
                newtrajectory[i] = (newtrajectory[i][0], newtrajectory[i][1],
                                    int(float(newtrajectory[i-1][2] + newtrajectory[i+1][2])/2.0))

    # Return the new trajectory list
    return newtrajectory


# Function: Sort trajectories by endpoint (y-coordinate)
def sortTrajectoriesByEndpoint(trajectories):
    """Sorts `trajectories` by the y-coordinate of their endpoint. Returns list of sorted trajectories or empty if
    an error occurred.
    """
    # Is trajectory a list?
    if type(trajectories) is not list:
        logger.error(u"sortTrajectoriesByEndpoint(trajectories): trajectories has to be a list.")
        return []

    # Sort by y values of endpoint
    trajectories.sort(key=lambda item: item[-1][1])

    # Return sorted list
    return trajectories


# Function: Calculate the weighted y-coordinate of a trajectory
def calculateWeightedYOfTrajectory(trajectory):
    """Calculates the weighted y-coordinate of `trajectory`. It sums up the weighted y-coordinates of every point in
    the trajectory. The weights are calculated on the distance from the endpoint: the closer to the endpoint the more
    it weights:

    `weight(point) = (endpoint.y - (endpoint.y - point.y) * (1.0 - reldistance(point, endpoint)))`

    Returns the weighted y-coordinate as `float`.
    """
    # Is trajectory a list?
    if type(trajectory) is not list:
        logger.error(u"calculateWeightedYOfTrajectory(trajectory): trajectories has to be a list.")
        return []

    # Weighted ys
    weightedy = []

    # Reference point is the last point (outlet); the further away a point is, the less weight it has and
    # the less it will add to the coordinate
    refpoint = (trajectory[-1][0], trajectory[-1][1])

    # Convert trajectory to a points array
    points = np.array([[x, y] for (x, y, w) in trajectory])

    # Sort the list by distance from refpoint
    points = sortCoordinatesByDistanceToPoint(points, (int(refpoint[0]), int(refpoint[1])))

    # The last point is now the furthest-away-point from the reference point; calculate the distance
    totaldistance = math.hypot(refpoint[0]-points[-1][0], refpoint[1]-points[-1][1])

    # Totaldistance is zero? Well...
    if totaldistance == 0.0:
        logger.warning(u"calculateWeightedYOfTrajectory(trajectory): totaldistance is zero.")
        return 0.0

    # Calculated the weighted y-coordinate for every point
    for point in points:
        # Relative distance of point from refpoint
        reldistance = (math.hypot(refpoint[0] - point[0], refpoint[1] - point[1]))/totaldistance

        # Calculate the weighted y-coordinate
        weightedy.append(refpoint[1] - (refpoint[1] - point[1]) * (1.0 - reldistance))

    # No weighted points?
    if len(weightedy) == 0:
        return 0.0

    # Return calculated weighted y-coordinate
    return float(sum(weightedy)/len(weightedy))


# Function: Sort trajectories by weighted y-coordinates of its points
def sortTrajectoriesByWeightedCoordinates(trajectories):
    """Sorts `trajectories` by their weighted y-coordinate (calculated by `ffe.calculateWeightedYOfTrajectory`).
    Returns list of sorted trajectories or empty if an error occurred.
    """
    # Is trajectory a list?
    if type(trajectories) is not list:
        logger.error(u"sortTrajectoriesByWeightedCoordinates(trajectories): trajectories has to be a list.")
        return []

    # Sort by y values of endpoint
    trajectories.sort(key=lambda item: calculateWeightedYOfTrajectory(item))

    # Return sorted list
    return trajectories


# Function: Get y and width of a trajectory as a function of x
def getYandWofTrajectory(trajectory, x):
    """Gets the y-coordinate and the width of `trajectory` as a function of `x` (interpolated by a spline).
    Returns a tuple in the form of `(y, w)`.
    """
    # Is trajectory a list?
    if type(trajectory) is not list:
        logger.error(u"getYandWofTrajectory(...): trajectory has to be a list.")
        return (0, -1)

    # We will be using cubic trajectories, so we need at least 3 points
    if len(trajectory) < 4:
        return (0, -1)

    # Get only x and y points of the trajectory
    points = np.array([(item[0], item[1]) for item in trajectory]).transpose()

    # Should x be outside of the boundary, return (0, -1)
    if not min(points[0]) <= x <= max(points[0]):
        return (0, -1)

    # Fit by spline
    spline = scipy.interpolate.InterpolatedUnivariateSpline(points[0], points[1], ext=3)

    # Get Y of spline(x)
    y = spline(x)

    # For the widths, we use the skin of the trajectory
    skin = getSkinOfTrajectory(trajectory)

    # Length should be a multiple of two
    if not len(skin) % 2 == 0:
        return (0, -1)

    # Length of half list
    halflen = int(len(skin)/2)

    # Separate into two lists, remove duplicates by dict, and invert the second one
    skinleft = np.array(dict(skin[:halflen-1]).items()).transpose()
    skinright = np.array(dict((skin[halflen:])[::-1]).items()).transpose()

    # Should x be outside of the boundary, return (Y, -1)
    if not min(skinleft[0]) <= x <= max(skinleft[0]) or not min(skinright[0]) <= x <= max(skinright[0]):
        return (y, -1)

    # This makes sure that points are in ascending order (increasing)!
    skinleft = skinleft[:, skinleft.argsort()[0]]
    skinright = skinright[:, skinright.argsort()[0]]

    # fit both with a spline; catch warnings
    with warnings.catch_warnings(record=True) as warn:
        splineleft = scipy.interpolate.UnivariateSpline(skinleft[0], skinleft[1], ext=3)
        splineright = scipy.interpolate.UnivariateSpline(skinright[0], skinright[1], ext=3)

        # If there is a warning, return (y, -1)
        if len(warn) > 0:
            return (y, -1)

    # The width is now the difference
    W = abs(splineleft(x)-splineright(x))

    # is W some NaN? Infinite?
    if math.isnan(W) or math.isinf(W):
        W = -1

    # Return values as tuple
    return (y, W)


# Function: Calculates the resolution of two trajectories as function of x
def getResolutionOfTrajectories(trajectory1, trajectory2, x):
    """Calculates the resolution between `trajectory1` and `trajectory2` as a function of `x`. Uses
    `ffe.getYandWofTrajectory` to get the y-coordinate and width at `x`.

    Returns the resolution as `float`.
    """
    # Are trajectories lists?
    if type(trajectory1) is not list or type(trajectory2) is not list:
        logger.error(u"getResolutionOfTrajectories(...): trajectories have to be lists.")
        return 0

    # Get x coordinates of trajectories
    xpoints1 = [item[0] for item in trajectory1]
    xpoints2 = [item[0] for item in trajectory2]

    # Is x outside of the boundaries of any of the trajectory? Then return zero.
    if not min(xpoints1) <= x <= max(xpoints1) or not min(xpoints2) <= x <= max(xpoints2):
        return 0

    # Get Y and W of first trajectory
    Y1, W1 = getYandWofTrajectory(trajectory1, x)

    # Get Y and W of second trajectory
    Y2, W2 = getYandWofTrajectory(trajectory2, x)

    # If both widths are zero, return zero
    if W2 == W1 == 0:
        return 0

    # If one of the widths is less than zero, return zero
    if W1 < 0 or W2 < 0:
        return 0

    # Calculate resolution
    res = float(abs(float(Y1-Y2))/(0.5*float(W1 + W2)))

    # Return resolution
    return res


