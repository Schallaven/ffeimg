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

from distutils.core import setup

setup(name="FFE",
      version="1.0",
      url="",
      author='Dr. Sven Kochmann',
      author_email='skochman@yorku.ca',
      py_modules=['ffe'],
      description="Frequently-used functions for free flow electrophoresis image processing",
      package_data={'ffe': ['ffe.m.html']})

