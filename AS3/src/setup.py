#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:03:07 2017

@author: ron
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("fastHough", ["fastHough.pyx"],
    include_dirs = [numpy.get_include()])

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})
