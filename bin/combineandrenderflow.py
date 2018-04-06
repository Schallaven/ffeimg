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
# combineandrenderflow.py (example)
#
# This quick&dirty example script reads the data from flowch1-5.csv, calculates the velocities, and plots it.
#

# Import modules
import cv2                          # OpenCV
import numpy as np                  # Numpy - You always need this.
import numpy.lib.recfunctions       # Some helper functions for numpy
import ffe                          # frequently-used function script
import sys                          # Sys functions
import logging                      # Logging functions
import getopt                       # Get and parse command-line arguments
import matplotlib.pyplot as plt     # For plotting intermediate graphs (debug)
import matplotlib.patches
import matplotlib.ticker
import scipy.stats                  # For the linear regression
import scipy.interpolate            # All these nice interpolation functions
import os                           # Some operating system functions
import math                         # Math!
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Input files (Experiment)
inputfiles = ['flowch' + str(i+1) for i in xrange(5)]

# Input file (COMSOL
inputcomsol = 'calculatedvelocityfield.txt'

# Reference image for physical properties, etc. (can be any image of the set)
refimage = 'flowch5/out0005.FFE.png'


# -----------------------------------------------------------------------------------------------------------------
# Read in data
# -----------------------------------------------------------------------------------------------------------------
# The whole data will go in here
data = np.array([], dtype=[('time', 'float64'), ('x', 'float64'), ('y', 'float64'), ('width', 'float64'),
                           ('vx', 'float64'), ('vy', 'float64')])

# Read each file (experiment)
for channel in inputfiles:
    # Print
    print("Processing %s ..." % channel)

    # Read the file
    channeldata = np.loadtxt(channel+'.csv', skiprows=1, dtype=[('time', 'float64'), ('x', 'float64'),
                                                         ('y', 'float64'), ('width', 'float64')])

    # No data read? :(
    if len(channeldata) == 0:
        continue

    # First of all, we use the very first line as reference time and change all time values according to this
    reftime = channeldata['time'][0]

    for i in xrange(len(channeldata['time'])):
        channeldata['time'][i] = channeldata['time'][i] - reftime

    # Create array for velocity vector
    velocities = np.zeros(len(channeldata)-1, dtype=[('vx', 'float64'), ('vy', 'float64')])

    # Calculate velocities
    for i in xrange(len(channeldata['x'])-1):
        # Time difference between current point and next point
        timediff = channeldata['time'][i+1] - channeldata['time'][i]

        # Zero time? This should not happen, so just skip for now
        if timediff == 0:
            print("Zero time :(")
            continue

        # Calculate velocity vector; unit is [mm/s]
        velocities[i][0], velocities[i][1] = (channeldata['x'][i+1]-channeldata['x'][i])/timediff, \
                                             (channeldata['y'][i+1]-channeldata['y'][i])/timediff

    # Remove the last line of data
    channeldata = np.delete(channeldata, -1)

    # Add velocities to channeldata
    channeldata = numpy.lib.recfunctions.merge_arrays((channeldata, velocities), flatten=True)

    # Add this channeldata to data
    data = np.concatenate((data, channeldata))


# Read COMSOL data
print("Processing %s ..." % inputcomsol)
comsoldata = np.loadtxt(inputcomsol, skiprows=8, dtype=[('x', 'float64'), ('y', 'float64'),
                                                        ('vx', 'float64'), ('vy', 'float64')])

# COMSOL velocity data is in meter per second, however we want mm per second, so...
comsoldata['vx'] *= 1000
comsoldata['vy'] *= 1000


# -----------------------------------------------------------------------------------------------------------------
# Prepare and interpolate experimental data for plotting
# -----------------------------------------------------------------------------------------------------------------
print("")
print("Preparing experimental data ...")

# Remove all almost-zero-velocities from the data (the part of the experiments
# in which the trajectory end does not move anymore)
#data = data[~np.logical_and(data['vx'] <= 0.01, data['vy'] <= 0.01)]

# Read data from reference image
ffedata = ffe.loadDictionaryFromPng(refimage)

# Data should include Inlets, Physical zone dimensions
if not "Inlets" in ffedata or not "Physical zone dimensions" in ffedata:
    print("FFE data from %s is not sufficient. Inlets and dimensions are missing." % refimage)

# Physical dimensions
physdim = ffedata["Physical zone dimensions"]

# Read outlets, if present
outlets = []
if "Outlets" in ffedata:
    outlets = ffedata["Outlets"]

# Let the origin be the first inlet if exists (origin is in real dimensions, i.e. mm!)
origin = (0, 0)

# First sort the points by distance from the center of the left side (0, zoneheight/2)
inlets = ffe.sortCoordinatesByDistanceToPoint(np.array(ffedata["Inlets"]), (0, int(physdim[1] / 2)))

# Calculate new origin from nearest point (first point in array)
origin = (inlets[0][0], inlets[0][1])

# Find minimum and maximum values for x and y; data only!
limit_x = (data['x'].min(), data['x'].max())
limit_y = (data['y'].min(), data['y'].max())

# Resolutions for the data point interpolation
rescoarse = 2.0
resfine = 0.1

# Generate the 2D grids for the coarse and fine data
y_coarse, x_coarse = np.mgrid[slice(-(physdim[1] - origin[1]), physdim[1]-origin[1], rescoarse),
                              slice(0, physdim[0] - origin[0], rescoarse)]

y_fine, x_fine = np.mgrid[slice(-(physdim[1] - origin[1]), physdim[1]-origin[1], resfine),
                          slice(0, physdim[0] - origin[0], resfine)]

# Interpolate experiment data: coarse velocity vectors
vx_coarse = scipy.interpolate.griddata((data['x'], data['y']), data['vx'], (x_coarse, y_coarse), method='nearest')
vy_coarse = scipy.interpolate.griddata((data['x'], data['y']), data['vy'], (x_coarse, y_coarse), method='nearest')
vl_coarse = np.empty_like(vx_coarse)

for i in xrange(len(vx_coarse)):
    for j in xrange(len(vx_coarse[i])):
        # Calculate the velocity value in [mm per s]
        vl_coarse[i, j] = math.hypot(vx_coarse[i, j], vy_coarse[i, j])

# Interpolate experiment data: fine velocity vectors
vx_fine = scipy.interpolate.griddata((data['x'], data['y']), data['vx'], (x_fine, y_fine), method='nearest')
vy_fine = scipy.interpolate.griddata((data['x'], data['y']), data['vy'], (x_fine, y_fine), method='nearest')
vl_fine = np.empty_like(vx_fine)

for i in xrange(len(vx_fine)):
    for j in xrange(len(vx_fine[i])):
        # Calculate the velocity value in [mm per s]
        vl_fine[i, j] = math.hypot(vx_fine[i, j], vy_fine[i, j])

# Print some data
print("")
print("Experimental data:")
print("average of x-component of velocity field is %.4f (std: %.4f)" % (np.average(data['vx']), np.std(data['vx'])))
print("average of y-component of velocity field is %.4f (std: %.4f)" % (np.average(data['vy']), np.std(data['vy'])))

print("Inter/Extrapolated data:")
print("average of x-component of velocity field is %.4f (std: %.4f)" % (np.average(vx_coarse), np.std(vx_coarse)))
print("average of y-component of velocity field is %.4f (std: %.4f)" % (np.average(vy_coarse), np.std(vy_coarse)))

# Export this data as one array
exportarray = []

for i in xrange(len(x_coarse.reshape(-1))):
    exportarray.append((x_coarse.reshape(-1)[i], y_coarse.reshape(-1)[i], vx_coarse.reshape(-1)[i],
                       vy_coarse.reshape(-1)[i]))

exportarray = np.array(exportarray, dtype=[('x [mm]', 'float64'), ('y [mm]', 'float64'),
                                           ('vx [mm/s]', 'float64'), ('vy [mm/s]', 'float64')])

# Export
np.savetxt('flowprofile.csv', exportarray, fmt='%.5f', delimiter='\t',
           header="\t".join(exportarray.dtype.names))

# -----------------------------------------------------------------------------------------------------------------
# Prepare and interpolate COMSOL data for plotting
# -----------------------------------------------------------------------------------------------------------------
print("")
print("Preparing COMSOL data ...")

# The origin of the COMSOL data is in the middle of the separation zone; let us move this to the origin
# little bit to the left;
comsoldata['x'] += float(physdim[0])/2.0 - origin[0]
comsoldata['y'] += float(physdim[1])/2.0 - origin[1]

# Interpolate experiment data: coarse velocity vectors
vx_comsol_coarse = scipy.interpolate.griddata((comsoldata['x'], comsoldata['y']), comsoldata['vx'],
                                              (x_coarse, y_coarse), method='nearest')
vy_comsol_coarse = scipy.interpolate.griddata((comsoldata['x'], comsoldata['y']), comsoldata['vy'],
                                              (x_coarse, y_coarse), method='nearest')
vl_comsol_coarse = np.empty_like(vx_comsol_coarse)

for i in xrange(len(vx_comsol_coarse)):
    for j in xrange(len(vx_comsol_coarse[i])):
        # Calculate the velocity value in [mm per s]
        vl_comsol_coarse[i, j] = math.hypot(vx_comsol_coarse[i, j], vy_comsol_coarse[i, j])

# Interpolate experiment data: fine velocity vectors
vx_comsol_fine = scipy.interpolate.griddata((comsoldata['x'], comsoldata['y']), comsoldata['vx'],
                                            (x_fine, y_fine), method='nearest')
vy_comsol_fine = scipy.interpolate.griddata((comsoldata['x'], comsoldata['y']), comsoldata['vy'],
                                            (x_fine, y_fine), method='nearest')
vl_comsol_fine = np.empty_like(vx_comsol_fine)

for i in xrange(len(vx_comsol_fine)):
    for j in xrange(len(vx_comsol_fine[i])):
        # Calculate the velocity value in [mm per s]
        vl_comsol_fine[i, j] = math.hypot(vx_comsol_fine[i, j], vy_comsol_fine[i, j])

# Print some data
print("")
print("COMSOL data:")
print("average of x-component of velocity field is %.4f (std: %.4f)" % (np.average(comsoldata['vx']),
                                                                        np.std(comsoldata['vx'])))
print("average of y-component of velocity field is %.4f (std: %.4f)" % (np.average(comsoldata['vy']),
                                                                        np.std(comsoldata['vy'])))

print("Inter/Extrapolated data:")
print("average of x-component of velocity field is %.4f (std: %.4f)" % (np.average(vx_comsol_coarse),
                                                                        np.std(vx_comsol_coarse)))
print("average of y-component of velocity field is %.4f (std: %.4f)" % (np.average(vy_comsol_coarse),
                                                                        np.std(vy_comsol_coarse)))

# Find the max velocity, so that both color bars are the same
colormax = vl_comsol_fine.max()
if vl_fine.max() > colormax:
    colormax = vl_fine.max()

# -----------------------------------------------------------------------------------------------------------------
# Start plotting
# -----------------------------------------------------------------------------------------------------------------
print("")
print("Plotting ...")
plt.figure(figsize=(15, 6))

# Create own color map
mycolormap = {'red': ((0.0, 0.59, 0.59),
                      (1.0, 0.00, 0.00)),

              'green': ((0.0, 0.86, 0.86),
                        (1.0, 0.63, 0.63)),

              'blue': ((0.0, 1.0, 1.0),
                       (1.0, 1.0, 1.0))
              }

light_dark_blue = matplotlib.colors.LinearSegmentedColormap('mycolormap', mycolormap)

# First subplot: COMSOL data
# ---------------------------
subfig1 = plt.subplot(121)

# Title and axes
plt.title('COMSOL', fontsize=18)
plt.xlabel('x [mm]', fontsize=18)
plt.ylabel('y [mm]', fontsize=18)

# Axis limits
plt.xlim(0, physdim[0] - origin[0])
plt.ylim(-(physdim[1] - origin[1]), physdim[1] - origin[1])

# Create a stream plot from the coarse data
colorplot = subfig1.pcolor(x_fine, y_fine, vl_comsol_fine, cmap=light_dark_blue, vmin=0, vmax=colormax)


# Creating a colorbar to the right of the plot, with a little padding
divider = make_axes_locatable(subfig1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(colorplot, cax=cax)
cbar.ax.set_ylabel(r'Velocity [$\mathregular{mm\;s^{-1}}$]', fontsize=18)
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

# Draw the measured data of the flow as vector arrows
subfig1.quiver(comsoldata['x'], comsoldata['y'], comsoldata['vx'], -comsoldata['vy'], color='red',
               zorder=3, linewidth=.5)

subfig1.streamplot(x_coarse, y_coarse, vx_comsol_coarse, vy_comsol_coarse, linewidth=1,
                   color='white', zorder=2)

# Add circles for inlets and outlets
for outlet in outlets:
    subfig1.add_artist(plt.Circle((outlet[0] - origin[0], outlet[1] - origin[1]), outlet[2], color='black'))

for inlet in inlets:
    subfig1.add_artist(plt.Circle((inlet[0] - origin[0], inlet[1] - origin[1]), inlet[2], color='black'))

# Set aspect ratio to 1:1
subfig1.set_aspect('equal')

# Invert y-axis
subfig1.invert_yaxis()


# Second subplot: Experimental data
# ---------------------------
subfig2 = plt.subplot(122)

# Title and axes
plt.title('Experiment', fontsize=18)
plt.xlabel('x [mm]', fontsize=18)
plt.ylabel('y [mm]', fontsize=18)

# Axis limits
plt.xlim(0, physdim[0] - origin[0])
plt.ylim(-(physdim[1] - origin[1]), physdim[1] - origin[1])

# Create a stream plot from the coarse data
colorplot = subfig2.pcolor(x_fine, y_fine, vl_fine, cmap=light_dark_blue, vmin=0, vmax=colormax)


# Creating a colorbar to the right of the plot, with a little padding
divider = make_axes_locatable(subfig2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(colorplot, cax=cax)
cbar.ax.set_ylabel(r'Velocity [$\mathregular{mm\;s^{-1}}$]', fontsize=18)
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

# Draw the measured data of the flow as vector arrows
subfig2.quiver(data['x'], data['y'], data['vx'], -data['vy'], color='red', zorder=3, linewidth=.5)

subfig2.streamplot(x_coarse, y_coarse, vx_coarse, vy_coarse, linewidth=1, color='white', zorder=2)

# Add circles for inlets and outlets
for outlet in outlets:
    subfig2.add_artist(plt.Circle((outlet[0] - origin[0], outlet[1] - origin[1]), outlet[2], color='black'))

for inlet in inlets:
    subfig2.add_artist(plt.Circle((inlet[0] - origin[0], inlet[1] - origin[1]), inlet[2], color='black'))

# Set aspect ratio to 1:1
subfig2.set_aspect('equal')

# Invert y-axis
subfig2.invert_yaxis()


# Tighten the layout
plt.tight_layout()

# Show the figure
plt.show()





