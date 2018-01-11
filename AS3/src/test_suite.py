#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:57:35 2017

@author: ron
"""

import datetime
import sys
import cv2
from main import drawLines, findLines, lineFit, colorPoints
import numpy as np
def main():
    """
    argv[1]: Image Path
    argv[2]: Hough Type
    argv[3]: Canny Threshold 1
    argv[4]: Canny Threshold 2
    argv[5]: Rho Resolution
    argv[6]: Theta Resolution
    argv[7]: Vote Threshold
    """
    if len(sys.argv) == 8:
        path = sys.argv[1]
        if sys.argv[2] == "-p":
            from houghTransform import houghTransform as ht
        elif sys.argv[2] == "-v1":
            from fastHough import fastHough as ht
        elif sys.argv[2] == "-v2":
            from fastHoughv2 import fastHoughv2 as ht
        c1 = int(sys.argv[3])
        c2 = int(sys.argv[4])
        r =  int(sys.argv[5])
        t =  int(sys.argv[6])
        h =  int(sys.argv[6])
    else:
        print("Not enough arguments. See /doc/CS512_AS3_Report.pdf\nExiting...")
        exit()

    img = cv2.imread(path) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                         
    edges = cv2.Canny(gray,c2,c1,apertureSize = 3)
    width, height = edges.shape
    accumulator, thetas, rhos, acc_votes = ht(edges,width,height,r,t,h)
    accumulator = np.asarray(accumulator)
    thetas = np.asarray(thetas)
    rhos = np.asarray(rhos)
    lines = findLines(acc_votes, thetas,rhos)
    img = drawLines(img,lines)
    lines = lineFit(acc_votes)
    img = drawLines(img,lines,(0,255,0))
    img = colorPoints(img,acc_votes)
    stamp = datetime.date.now + " " + sys.argv[2]+".jpg"
    cv2.imwrite(stamp,img)