# Image processing and analysis system for development and use of free flow electrophoresis chips

## Introduction

This image processing and analysis system is designed to facilitate detailed performance analysis of free flow electrophoresis (FFE) chips. It consists of a comprehensive customizable software suite which complement the self-built hardware (see paper below). All components were designed modularly to be accessible, adaptable, versatile, and automatable. 

The system provides tools for i) automated identification of chip features (e.g. separation zone and flow markers), ii) extraction and analysis of stream trajectories, and iii) evaluation of flow profiles and separation quality (e.g. determination of resolution). Equipped with these tools, the presented image processing and analysis system will enable faster development of FFE chips and applications. It will also serve as a robust detector for fluorescence-based analytical applications of FFE.

## Paper
These scripts where published first in [Lab on a Chip](http://pubs.rsc.org/en/content/articlehtml/2016/lc/c6lc01381c).

## Software

This software suite includes a function library and some ready-to-use programs for setting up and aligning the system and the chip as well as recording, storing, and processing images and extracting performance parameters from these images. The programs were written in Python 2.7 according to Eric Raymond's 17 Unix Rules (in particular, Rules of Simplicity and Modularity). Main modules used are Numpy, Mathplotlib, Scipy, and OpenCV (Open Source Computer Vision Library). The source code as well as full documentation can be found in the ESI.† The software suite is compatible with every camera supported by OpenCV including a variety of web-cameras (see [documentation of OpenCV](http://docs.opencv.org/2.4.13/modules/refman.html)).

## Contributions

Contributions are welcome!

