#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:19 2017

@author: ron
"""


import cv2
import numpy as np
from fastHoughv2 import fastHoughv2
        
def nothing(x):
    pass

def findLines(acc_votes, thetas,rhos):
    lines=[]
    for i,j in acc_votes.keys():
        lines.append((rhos[i],thetas[j]))
    return lines

def drawLines(image,lines,color=(0,0,255)):
    for rho,theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),color,2)
    return image

def lineFit(acc_votes):
    lines = []
    for key in acc_votes:
        b = np.array(acc_votes[key])
        vx,vy,x,y = cv2.fitLine(b, cv2.DIST_L2,0,0.01,0.01)
        alpha = [x[0],y[0]]
        beta = [vx[0],vy[0]]
        r = np.abs(np.cross(alpha,beta))
        t = np.pi/2 - np.arccos(vx[0])
        lines.append((r,t))
    return lines

def colorPoints(image,acc_votes):
    for k,v in acc_votes.items():
        for x,y in v:
            image[x,y] = np.array([0,126,126],dtype=np.uint8)
    return image
    

def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Image",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TrackBars",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Canny Threshold 1","TrackBars",200,200,nothing)
    cv2.createTrackbar("Canny Threshold 2","TrackBars",100,200,nothing)
    cv2.createTrackbar("Hough Threshold","TrackBars",200,200,nothing)
    cv2.createTrackbar("Theta resolution","TrackBars",1,15,nothing)
    cv2.createTrackbar("Rho resolution","TrackBars",1,250,nothing)
    cv2.namedWindow("Edges",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Hough Plane",cv2.WINDOW_AUTOSIZE)
    while True:
        print("Processing....")
        frame, image = camera.read()
        if not frame:
            break
            cv2.destroyAllWindows()
        cv2.imshow("Image",image)
        edges = cv2.Canny(image,cv2.getTrackbarPos("Canny Threshold 1","Trackbars"),
                          cv2.getTrackbarPos("Canny Threshold 1","Trackbars"))
        cv2.imshow("Edges",edges)
        w,h = edges.shape
        accumulator, thetas, rhos, acc_votes = fastHoughv2(edges,w,h,
                                                           cv2.getTrackbarPos("Rho Resolution","Trackbars"),
                                                           cv2.getTrackbarPos("Theta Resolution","TrackBars"),
                                                           cv2.getTrackbarPos("Threshold","TrackBars"))
        accumulator = np.asarray(accumulator)
        thetas = np.asarray(thetas)
        rhos = np.asarray(rhos)
        lines = findLines(acc_votes,thetas,rhos)
        image = drawLines(image,lines)
        lines = lineFit(acc_votes)
        image = drawLines(image,lines,(0,255,0))
        image = colorPoints(image,acc_votes)
        accumulator = np.transpose((accumulator/np.max(accumulator)*255))
        cv2.imshow("Image",image)
        cv2.imshow("Hough Plane",accumulator)
        k = cv2.waitKey(0)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break
        else:
            continue


if __name__ == "__main__":
    main()