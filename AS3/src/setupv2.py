#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:26:11 2017

@author: ron
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("fastHoughv2", ["fastHoughv2.pyx"],
    include_dirs = [numpy.get_include()])

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})