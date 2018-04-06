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
# Acquire (Automated console/GUI program)
#
# This program records snapshots and a video using the camera settings and measurement settings . It is
# automated and only provides a live view with no user interaction (except for [q]uit).
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

# Step 0: Setup logging
# --------------------------------------------------------------------------------------------------------------------
# Create logger object for this script
logger = logging.getLogger('FFE')

# Set level of information
logger.setLevel(logging.DEBUG)

# Create log file handler which records everything
fh = logging.FileHandler('recording.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t%(filename)s\t%(levelname)s\t%(lineno)d\t%(message)s')
fh.setFormatter(formatter)

# Create console handler which shows INFO and above (WARNING, ERRORS, CRITICALS, ...)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)

# Add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# Start the program
logger.info(u"Starting new record session")


# Step 1: Open camera, read and apply camera settings, try to read one frame to see if it is working
# --------------------------------------------------------------------------------------------------------------------
logger.info(u"Open camera device, apply, and load recording settings")

# Log and save camera ID
cameraid = ffe.getCameraID()
logger.info(u"Camera ID used: %d", cameraid)

# Open camera device
camera = cv2.VideoCapture(cameraid)

# Apply camera settings; load recording settings
camera_settings = ffe.applyCameraSettings(camera)

# Log camera settings
ffe.logCameraProps(camera)

# Get frame width and height
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Try to get very first frame from camera
ret, image = camera.read()

# Did not work?
if not ret:
    logger.error(u"Could not read frame from camera. Is it connected?")
    sys.exit(1)

logger.info(u"Camera is working")

# Log recording settings
ffe.logRecordingSettings(camera_settings)


# Step 2: Read output settings and setup folder
# --------------------------------------------------------------------------------------------------------------------
# Read settings
output_settings = ffe.getOutputSettings()

# Log
ffe.logOutputSettings(output_settings)

# Generate base name for folder ("Datetime stamp - measurement name")
folderbasename = time.strftime('%Y-%m-%d %H-%M') + ' - ' + output_settings["name"]

# Rootdir given?
if len(output_settings["rootdir"]) > 0:
    folderbasename = output_settings["rootdir"] + '/' + folderbasename

# Initial folder name is only the basename
foldername = folderbasename

# Generate actual folder name (folderbasename + 001) and check if it exists
fn = 1
while True:
    # Check if the folder DOESN'T exists, if yes create the folder and get out
    if not os.path.exists( foldername ):
        # Try to create the folder
        try:
            os.makedirs(foldername)
        except os.error as error:
            logger.error(u"Could net create folder '%s', although it does not exist?! OS says: %s", foldername, format(error))
            sys.exit(2)
        # Get out
        break

    # Other wise, let us add a number to foldernamebasename
    foldername = folderbasename + ' - ' + ('%03d' % fn)

    # Otherwise, increase number at the end
    fn += 1

logger.info(u"Created folder '%s'", foldername)


# Step 3: Record simple background
# --------------------------------------------------------------------------------------------------------------------
logger.info(u"Trying to record background image")
# Flush log before using raw_input
[h.flush() for h in logger.handlers]

# Create empty array for the background
background = np.zeros((frame_height, frame_width, 3), np.uint8)
backgroundshow = np.zeros((frame_height, frame_width, 3), np.uint8)

while True:
    raw_input(u"This program now will record a background image. Please turn OFF the excitation light "
              u"and press ENTER to continue.")

    # Wait 3 seconds; sometimes when the user just turned off the excitation light and the background
    # images is recorded, the image is full of artifacts
    time.sleep(3)

    # Record the frame
    ret, background = camera.read()

    # Flipping?
    if camera_settings["doflipping"]:
        backgroundshow = cv2.flip(background, camera_settings["flip"])
    else:
        backgroundshow = background

    # Show the flipped image to the user
    cv2.imshow(u"Is this background image ok? Press y to confirm, q for exit, or any other key to retry", backgroundshow)

    # Keys
    key = cv2.waitKey(0) & 0xFF

    if key == ord('y'):
        break

    if key == ord('q'):
        logger.info(u"User decided to quit while taking background image.")
        sys.exit(0)

# Close windows
cv2.destroyAllWindows()

# Save background image shown to the user
cv2.imwrite(foldername + '/background.png', backgroundshow)

logger.info("Recorded and saved background image.")


# Step 4: Setup video recording
# --------------------------------------------------------------------------------------------------------------------
logger.info("Starting video recorder")

# Generate fourCC code from codec
fourcc = int(cv2.VideoWriter_fourcc(*output_settings["videocodec"]))

# Initialize video writer
videorecorder = cv2.VideoWriter(foldername + '/output.avi', fourcc, int(output_settings["videofps"]), (frame_width,frame_height), True)

# Something wrong?
if not videorecorder.isOpened:
    logger.error(u"Video recorder could not be opened (FourCC: %d)", int(fourcc))
    sys.exit(1)


# Step 5: Start recording
# --------------------------------------------------------------------------------------------------------------------
logger.info("Start recording")

# Create a frame that will hold the integrated images (32bit instead of the 8bit per channel!)
integratedframe = np.zeros((frame_height, frame_width, 3), np.uint32)

# Create an empty presentation frame, which is then updated occasionally
presentationframe = np.zeros((frame_height, frame_width, 3), np.uint8)

# Frame counter
framecounter = 0

# Integrated frame counter
integratedframecounter = 0

# Image counter
imagecounter = 1

# Save start of measuring time
measurementstart = time.clock()

# Latest time point
latesttime = measurementstart

# Record loop
# In this script time.sleep is used; if more precision (more images per second) is needed threads or more precise
# timers could be used;
while True:
    # Read a frame from the camera
    ret, frame = camera.read()

	# Flipping?
    if camera_settings["doflipping"]:
        frame = cv2.flip(frame, camera_settings["flip"])
 
    # Remove background from image
    if int(output_settings["ignorebackground"]) == 0:
        frame = cv2.subtract(frame, background)

    # Add frame to integration
    integratedframe = np.add(integratedframe, frame)

    # Increase frame counter
    framecounter += 1

    # Elapsed time
    elapsed = time.clock() - latesttime

    # Elapsed time larger than integration time?
    if elapsed > float(camera_settings["integratetime"]):
        # Split the frame (cv2.split does not support 32bit arrays)
        blue, green, red = integratedframe[:, :, 0], integratedframe[:, :, 1], integratedframe[:, :, 2]

        # Only use channels, which were selected by the user; otherwise replace by empty channel
        if 'blue' not in camera_settings:
            blue = np.zeros((frame_height, frame_width), np.uint32)
        if 'green' not in camera_settings:
            green = np.zeros((frame_height, frame_width), np.uint32)
        if 'red' not in camera_settings:
            red = np.zeros((frame_height, frame_width), np.uint32)

        # Individual scaling?
        if int(output_settings["scaleindividually"]) == 1:
            blue = ffe.rescaleFrameTo8bit(blue)
            green = ffe.rescaleFrameTo8bit(green)
            red = ffe.rescaleFrameTo8bit(red)

        # Combine channels?
        if int(output_settings["combinechannels"]) == 1:
            # Put everything in the green channel
            integratedframe[:, :, 0], integratedframe[:, :, 1], integratedframe[:, :, 2] = 0, blue + green + red, 0
        else:
            # Otherwise, just remerge (this is important in case the channels were rescaled!)
            # Again, cv2.merge does not support 32bit frames
            integratedframe[:, :, 0], integratedframe[:, :, 1], integratedframe[:, :, 2] = blue, green, red

        # Rescale and convert to 8bit image again
        integratedframe = ffe.rescaleFrameTo8bit(integratedframe)

        # Timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Time difference (relative time since starting of measurement)
        timedf = time.clock() - measurementstart

        # Prepare filedata
        filedata = {"Experiment name": output_settings["name"], "Dataline 1": output_settings["dataline1"],
                    "Dataline 2": output_settings["dataline2"], "Channels recorded": camera_settings["allchannels"],
                    "Timestamp": timestamp, "Snapshot time": timedf, "Image counter": imagecounter,
                    "Total frame counter": framecounter}

        # Prepare overlay image (green on black is best)
        overlayimage = ffe.prepareOverlayImage((frame_width, frame_height), (0, 255, 0), filedata)

        # Create a presentation picture for showing to user and for writing to videofile
        presentationframe = cv2.add(integratedframe, overlayimage)

        logger.info("%d frames acquired at %.1fs", framecounter, timedf)

        # Counter
        integratedframecounter += 1

        # Want a snapshot frame?
        if (integratedframecounter % int(output_settings["snapshots"]) == 0):
            # Filename (style: out4321.FFE.png)
            filename = foldername + '/out%04d.FFE.png' % imagecounter

            # Save image to a png first
            cv2.imwrite(filename, integratedframe)

            # Add extra chunk data to file
            ffe.updateDictionaryOfPng(filename, filedata)

            # Add presentation frame to video file
            videorecorder.write(presentationframe)

            # Logging
            logger.info("Frame was saved as out%04d.FFE.png and in video file", imagecounter)

            # Add image counter
            imagecounter += 1

        # Start with blank frame
        integratedframe = np.zeros((frame_height, frame_width, 3), np.uint32)

        # Set latesttime to current time
        latesttime = time.clock()

    # Show presentationframe
    cv2.imshow("Measurement; Press q for quit", presentationframe)

    # Keys
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break


# Last step: cleaning up and copying the log file
# --------------------------------------------------------------------------------------------------------------------
# Release and close the camera
camera.release()

# Release and close the video recorder
videorecorder.release()

# Destroy all windows
cv2.destroyAllWindows()

# Final logging
logger.info("Recording session ended")

# Copying log file to measurement folder
shutil.copyfile("recording.log", foldername + "/recording.log")




