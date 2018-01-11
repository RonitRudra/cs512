#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:34:38 2017

@author: ron
"""

import numpy as np
import cv2
import os
import sys

ref_left =[]
ref_right = []
N = 50
colors = []
for i in range(N):
    colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))


###############################################################################
########################### Drawing Functions #################################

def readImagePairs(name):
    filenames = os.listdir("../data/")
    left, right = name+"-l.tif",name+"-r.tif"
    if left in filenames and right in filenames:
        imleft = cv2.imread(os.path.join("../data/",left))
        imright = cv2.imread(os.path.join("../data/",right))
        return imleft, imright
    else:
        print("Image(s) not found.")
        return None, None
    
def drawRef(x,y,param):
    # X and Y coordinates are messed up for some reason
    if param == "left":
        index = len(ref_left) - 1
        cv2.circle(main.im_left,(y,x),5,colors[index],2)
    if param == "right":
        index = len(ref_right) - 1
        cv2.circle(main.im_right,(y,x),5,colors[index],2)
    
def _click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == "left":
            ref_left.append((y,x))
        if param == "right":
            ref_right.append((y,x))
        drawRef(y,x,param)
 
def _draw(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == "left":
            cv2.circle(main.im_left,(x,y),5,(0,0,255),2)
            drawEpipolarLine(y,x,param)
        if param == "right":
            cv2.circle(main.im_right,(x,y),5,(0,0,255),2)
            drawEpipolarLine(y,x,param)
       
def drawEpipolarLine(x,y,flag):
    pt = np.array([x,y,1])
    if flag == "left":
        r,c,d = main.im_right.shape
        r = np.dot(main.F,pt)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        main.im_right = cv2.line(main.im_right, (y0,x0), (y1,x1),(0,255,0),1)
    else:
        r,c,d = main.im_left.shape
        r = np.dot(main.F.T,pt)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        main.im_left = cv2.line(main.im_left, (y0,x0), (y1,x1),(0,255,0),1)

###############################################################################
        
def homogenize(points):
    """
    Homogenizes a set of points
    [PARAMETERS]:
        points (list): A list of 2-tuples of point coordinates
    [RETURNS]:
        p (ndarray): An array of homogeneous points
    """
    p = np.array(points)
    p = np.append(p,np.ones(p.shape[0]).reshape(-1,1),axis=1) 
    
    return p
    

def normalize(points):
    """
    Calculates normalized image point coordinates.
    [PARAMETERS]:
        points (list): An n-length list containing 2-tuples of coordinates
    [RETURNS]:
        points_norm (ndarray): An nx3 array of normalized homogeneous coordinates
        M_norm (ndarray): The normalization matrix
    """
    # points is an n length list of two-tuples
    p = np.array(points)
    mu_x = np.mean(p[:,0])
    mu_y = np.mean(p[:,1])
    sigma = np.std(p)
    
    S = np.array([[1/sigma,0,0],[0,1/sigma,0],[0,0,1]])
    T = np.array([[1,0,-mu_x],[0,1,-mu_y],[0,0,1]])
    M = np.matmul(S,T)
    points_norm = []
    for x,y in p:
        q = np.matmul(M,np.array([x,y,1]).T)
        points_norm.append(q)
    
    return np.array(points_norm), M
       
def calcFundamental(p_l,p_r,M_l,M_r):
    """
    Calculates Fundamental Matrix using the 8-Point Algorithm.
    [PARAMETERS]:
        p_l (ndarray): Array of n normalized homogeneous points of left image
        p_r (ndarray): Array of n normalized homogeneous points of right image
        M_l (ndarray): A 3x3 normalization matrix for left image
        M_r (ndarray): A 3x3 normalization matrix for right image
    [RETURNS]:
        F (ndarray): A 3x3 Fundamental Matrix with last element 1
    """
    n = p_l.shape[0]
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [p_l[i,0]*p_r[i,0], p_l[i,0]*p_r[i,1], p_l[i,0]*p_r[i,2],
                p_l[i,1]*p_r[i,0], p_l[i,1]*p_r[i,1], p_l[i,1]*p_r[i,2],
                p_l[i,2]*p_r[i,0], p_l[i,2]*p_r[i,1], p_l[i,2]*p_r[i,2] ]
    # SVD decomposition of A
    U,d,V = np.linalg.svd(A)
    # Find Right Null Space of A which is F
    # Using V[-1] for last row as SVD returns V.T as V
    F = V[-1].reshape(3,3)
    if np.linalg.matrix_rank(F) != 2:
        # F is not necessarily rank 2
        U,d,V = np.linalg.svd(F)
        # Make F rank 2 by setting last element of d = 0
        d[2] = 0
        # Recompute F using new d
        F = np.dot(U,np.dot(np.diag(d),V))
    # Undo normalization
    F = np.dot(M_l.T,np.dot(F,M_r))
    F = F/F[2,2]
    
    return F

def calcEpipole(F):
    """
    Calculates the left and right epipole from Fundamental matrix
    [PARAMETERS]:
        F (ndarray): A 3x3 Fundamental Matrix with last element 1
    [RETURNS]:
        e_l (ndarray): A homogeneous point for left epipole
        e_r (ndarray): A homogeneous point for right epipole
    """
    
    # left epipole is right null space of F
    # right epipole is left null space of F
    U,S,V = np.linalg.svd(F)
    el = V[-1]
    er = U.T[-1]
    return el/el[2], er/er[2]

def showHelp():
    f = open("../doc/help.txt")
    x = f.read()
    f.close()
    print(x)

def main():
    if len(sys.argv) != 2:
        raise ValueError("Missing Arguments")
    filename = sys.argv[1]
    im_left, im_right = readImagePairs(filename)
    
    if im_left is None and im_right is None:
        raise ValueError("File does not exist. Exiting...")
    
    main.im_left, main.im_right = im_left, im_right
    
    showHelp()
    x = input("Press any key to continue:")
    
    cv2.namedWindow("Left Camera",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Right Camera",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Left Camera",_click,"left")
    cv2.setMouseCallback("Right Camera",_click,"right")
   
    while True:
        cv2.imshow("Left Camera",main.im_left)
        cv2.imshow("Right Camera",main.im_right)
        print("Left points:")
        print(ref_left)
        print("Right points:")
        print(ref_right)
        k = cv2.waitKey(5)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break
    
    assert len(ref_left) == len(ref_right), "Unequal Number of Features Selected."
    k = input("Press 'c' to continue or 'e' to exit:")
    if k == 'e':
        exit()
        
    P_l, M_l = normalize(ref_left)
    P_r, M_r = normalize(ref_right)
    F = calcFundamental(P_l,P_r,M_l,M_r)
    main.F = F
    el,er = calcEpipole(main.F)
    
    print("Matrix F:")
    print(F)
    print("Left Epipole:")
    print(el)
    print("Right Epipole:")
    print(er)
    
    main.im_left, main.im_right = readImagePairs(filename)
    cv2.namedWindow("Left Camera",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Right Camera",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Left Camera",_draw,"left")
    cv2.setMouseCallback("Right Camera",_draw,"right")
    

    
    while True:
        cv2.imshow("Left Camera",main.im_left)
        cv2.imshow("Right Camera",main.im_right)
        k = cv2.waitKey(5)
        if k == ord('e'):
            cv2.destroyAllWindows()
            break
        
    
if __name__ == "__main__":
    main()