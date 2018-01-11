#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:58:19 2017

@author: ron
"""

import sys
import numpy as np
from main import calcEpipole, calcFundamental, normalize, homogenize

def main():
    if len(sys.argv) != 2:
        raise ValueError("Missing Argument")
        
    n = int(sys.argv[1])
    
    f=open('../data/matched.txt')
    data = f.read()
    f.close()
    
    data = data.split('\n')
    ref_left = []
    ref_right = []
    for i,d in enumerate(data):
        x = d.split()
        ref_left.append((int(x[0]),int(x[1])))
        ref_right.append((int(x[2]),int(x[3])))
        if i == n-1:
            break
    
    print(ref_left)
    print(ref_right)
    
    P_l, M_l = normalize(ref_left)
    P_r, M_r = normalize(ref_right)
    p_l = homogenize(ref_left)
    p_r = homogenize(ref_right)
    F = calcFundamental(P_l,P_r,M_l,M_r)
    el,er = calcEpipole(F)
    print("Matrix F:")
    print(F)
    print("Rank of F: %d"%np.linalg.matrix_rank(F))
    print("Left Epipole:")
    print(el)
    print("Right Epipole:")
    print(er)
    print("Epipolar Constraint RHS:")
    for a,b in zip(p_l,p_r):
        print(np.dot(b,np.dot(F,a)))

    print
        
        
if __name__=="__main__":
    main()
        
    