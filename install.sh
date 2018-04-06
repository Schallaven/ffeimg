#!/bin/bash

# This script creates soft links in ~/bin to the scripts in the
# bin directory of this project. The links will have the form of
# ffe-scriptname, e.g. ffe-pngdata for pngdata.py.

# Save the current directory in a variable, so that we don't call pwd
# over and over again
CURDIR=$(pwd)

# Check if ~/bin exists and is a directory; if not create it
[ ! -d "$HOME/bin" ] && mkdir "$HOME/bin"

# Create all the links we need (not every script has to be linked)
ln -s "$CURDIR/bin/pngdata.py" "$HOME/bin/ffe-pngdata"
ln -s "$CURDIR/bin/acquire.py" "$HOME/bin/ffe-acquire"
ln -s "$CURDIR/bin/alignchip.py" "$HOME/bin/ffe-alignchip"
ln -s "$CURDIR/bin/findfeatures.py" "$HOME/bin/ffe-findfeatures"
ln -s "$CURDIR/bin/findtrajectories.py" "$HOME/bin/ffe-findtrajectories"
ln -s "$CURDIR/bin/rendertrajectories.py" "$HOME/bin/ffe-rendertrajectories"
ln -s "$CURDIR/bin/resolution.py" "$HOME/bin/ffe-resolution"
ln -s "$CURDIR/bin/setupcamera.py" "$HOME/bin/ffe-setupcamera"
ln -s "$CURDIR/bin/setupmeasurement.py" "$HOME/bin/ffe-setupmeasurement"

