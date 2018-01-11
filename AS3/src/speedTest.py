#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:01:14 2017

@author: ron
"""

from houghTransform import houghTransform
from fastHough import fastHough
from fastHoughv2 import fastHoughv2
from timeit import default_timer as timer
from cv2 import imread, cvtColor, Canny, COLOR_BGR2GRAY
import numpy as np

def main():
    img = imread("../data/dave.jpg") 
    gray = cvtColor(img,COLOR_BGR2GRAY)                         
    edges = Canny(gray,50,150,apertureSize = 3)
    width, height = edges.shape
    
    print("Running Python Hough Transform..\n")
    t = timer()
    accumulator, thetas, rhos, acc_votes = houghTransform(edges,1,1,200)
    py = timer()-t
    print("Execution completed in %f seconds.\n"%(py))
    print("Running Cython Hough Transform..\n")
    t = timer()
    accumulator, thetas, rhos, acc_votes = fastHough(edges,width,height,1,1,200)
    cy = timer()-t
    print("Execution completed in %f seconds.\n"%(cy))
    print("Cython v1 code is %.3f times faster"%(py/cy))
    print("Running Cython Hough Transform Version 2..\n")
    t = timer()
    xi,yi = np.nonzero(edges)
    accumulator, thetas, rhos, acc_votes = fastHoughv2(edges,width,height,1,1,200)
    cy = timer()-t
    print("Execution completed in %f seconds.\n"%(cy))
    print("Cython v2 code is %.3f times faster"%(py/cy))
    
if __name__ == '__main__':
    main()
    
    