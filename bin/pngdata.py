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
# Pngdata (console program)
#
# This program allows viewing and editing the dfFe chunk in a PNG file on the command line. Can be used to
# add information about features to the file for evaluation.
#

# Import modules
import ffe              # frequently-used function script
import sys              # Sys functions
import logging          # Logging functions
import png              # Raw PNG read/write functions
import getopt           # Get and parse command-line arguments


# Function: Prints help page
# --------------------------------------------------------------------------------------------------------------------
def printHelpPage():
    print("USAGE: script.py [options] --input-file <file>")
    print("")

    print("Order of options is not important. The input file is mandatory.")
    print("")

    print("Switches:")
    print("\t--help:\t\t\t\t\tShows this help page.")
    print("")
    print("Options:")
    print("\t--command <command>\t\tCommand to perform. Default: show. Some commands require extra "
          "parameters. See below for more information.")
    print("")
    print("Commands:")
    print("\tshow\t\t\t\t\tShows contents of dfFe-chunk data as table.")
    print("\tcopy\t\t\t\t\tCopies data from an input file (--input-file) to an output file (--output-file). "
          "Only copies evaluated data. To copy everything specify --copyall switch.")
    print("\tchunks\t\t\t\t\tShows information about the chunks in the file.")
    print("\tadd\t\t\t\t\t\tAdds/updates data in the PNG file. The command reads from the standard input. Each line "
          "represents a data line in one of two forms: \"<key>=<data>\", with <key> and <data> being simple strings. "
          "Alternatively, you can use a double equal sign: \"<key>==<data>\", with <key> being a simple string and "
          "<data> being evaluated by Python. This way, you can add lists, tuples, dictionaries, etc to the data chunk "
          "of the PNG file.")
    print("\tdel\t\t\t\t\t\tRemoves data from the input file. The command reads from the standard input. Each line "
          "represents a key to delete from the file. Will not delete acquisition data unless --forcedel is provided.")
    print("")


# Step 0: Parse command-line arguments
# --------------------------------------------------------------------------------------------------------------------
# Inputfile
inputfile = ""
outputfile = ""

# This is the list of keys, which are recorded during acquisition
acquisitionkeys = ["Experiment name", "Dataline 1", "Dataline 2", "Channels recorded", "Timestamp",
                   "Snapshot time", "Image counter"]

# Command, default: show content of dictionary
command = "show"

# Parameters
copyall = False
forcedel = False

# No parameters given?
if len(sys.argv) == 1:
    printHelpPage()
    sys.exit(1)

# Try to find all the arguments
try:
    # All the options to recognize
    opts, args = getopt.getopt(sys.argv[1:], "", ["help", "input-file=", "output-file=", "command=",
                                                  "copyall", "forcedel"])

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
    elif opt == "--command":
        if arg in ["show", "copy", "chunks", "add", "del"]:
            command = arg
        else:
            print("Command '%s' not recognized. Please see help page." % str(arg))
            sys.exit(0)
    elif opt == "--copyall":
        copyall = True
    elif opt == "--forcedel":
        forcedel = True


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

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# For certain commands such as "show" we do not want to log anything
if command in ["show", "chunks"]:
    # Practically, we just set the logger to a very high level
    logger.setLevel(logging.CRITICAL+1)

# Start the program
logger.info(u"####### Manipulating PNG data #######")


# Step 1: Open file and read dfFe content as dictionary
# --------------------------------------------------------------------------------------------------------------------
logger.info(u"Open and read file '%s'", inputfile)

# Read data
ffedata = ffe.loadDictionaryFromPng(inputfile)


# Step 2: Execute commands
# --------------------------------------------------------------------------------------------------------------------

# Show content of ffedata (NO LOGGING)
if command == "show":
    # No content?
    if len(ffedata) == 0:
        print("There is no dfFe chunk in the file.")
    # Else, show content of dfFe in table
    else:
        print("")
        print("{:<30} {:<30}".format('Key', 'Data'))
        print("-"*80)
        # Print the sorted data
        for key, data in sorted(ffedata.iteritems()):
            # String
            data = str(data)
            # Shorten the data if necessary
            dataline = (data[:45] + ' (..)') if len(data) > 50 else data
            # Print
            print("{:<30} {:<30}".format(key, dataline))
        print("")

    # End of operation
    sys.exit(0)

# Show information about chunks in file
elif command == "chunks":
    # Open the file
    pngfile = png.Reader(inputfile)

    # Get list of chunks in file
    chunklist = list(pngfile.chunks())

    # Print chunk data
    print("")
    print("{:<6} {:<8} {:<50}".format('Chunk', 'Length', 'Data'))
    print("-" * 80)
    for data in chunklist:
        # Shorten the string
        dataline = (data[1][:45] + ' (..)') if len(data[1]) > 50 else data[1]
        # Remove newline characters
        dataline = dataline.translate(None, "\r\n")
        print("{:<6} {:<8} {:<50}".format(data[0], len(data[1]), dataline))

    print("")

# Copy data from one file to another (LOGGING)
elif command == "copy":
    # No output file given?
    if len(outputfile) == 0:
        logger.error(u"No outpfile specified for copy command.")
        sys.exit(1)

    # Logging
    logger.info(u"Copy data from '%s' to '%s'.", inputfile, outputfile)

    # Copy dictionary
    outputdata = {}

    # If not suppressed we want to overwrite only some data
    if not copyall:
        logger.info(u"Will not overwrite data recorded during acquisition.")
        # Filter the acquisition keys from input data
        outputdata = {key:value for key, value in ffedata.iteritems() if key not in acquisitionkeys}

    # Else: Copy everything
    else:
        logger.info(u"Copy everything from input file to output file.")
        outputdata = ffedata

    # Update data in output-file
    ffe.updateDictionaryOfPng(outputfile, outputdata)

    # End of operation
    sys.exit(0)

# add data from input lines to input file (LOGGING)
elif command == "add":
    logger.info(u"Read from standard input.")

    # Create empty dictionary
    newdata = {}

    # Read from stdin (this could also be a file piped in with <<)
    lines = sys.stdin.readlines()
    for line in lines:
        # Strip string of new line, tab, or space characters at end or in the beginning
        line = line.strip()

        # Ignore empty lines
        if len(line) == 0:
            continue

        # No equal sign? Ignore!
        if line.find("=") == -1:
            continue

        # Evaluate data?
        evaluatedata = bool(line.find("==") != -1)

        # Split line into two things
        key, data = line.split("=", 1)[0], line.split("=", 1)[1]

        # Evaluated? Then eat first byte of data (== "=", because of the double "==")
        if evaluatedata:
            data = data.lstrip("=")

        # Update data in new dictionary; if we evaluate the data then strip all "builins" such as function calling
        # Here it is less about security then about breaking-the-script
        newdata.update({key: eval(data, {'__builtins__': {}}) if evaluatedata else data})

    # Update data in file
    ffe.updateDictionaryOfPng(inputfile, newdata)

    logger.info("Items added. Dictionary of '%s' was updated.", inputfile)

# del data from input file (LOGGING)
elif command == "del":
    logger.info(u"Read from standard input.")

    # Read from stdin (this could also be a file piped in with <<)
    for line in sys.stdin:
        # Strip string of new line, tab, or space characters at end or in the beginning
        line = line.strip()

        # Ignore empty lines
        if len(line) == 0:
            continue

        # Do not delete acquisition keys if not --force-del is provided
        if not forcedel and line in acquisitionkeys:
            logger.warning(u"Will not delete acquisition key '%s' because --force-del is not provided.", line)
            continue

        # Try to delete key from dictionary (if exists, otherwise, ignore)
        if line in ffedata:
            del ffedata[line]

    # Update data in file
    ffe.replaceDictionaryOfPng(inputfile, ffedata)

    logger.info(u"Items removed. Dictionary of '%s' was updated.", inputfile)



