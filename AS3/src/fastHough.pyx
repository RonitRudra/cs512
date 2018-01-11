#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:48:59 2017

@author: ron
"""

import numpy as np
from collections import defaultdict

def fastHough(unsigned char[:,:] image, int width, int height, int rho, int theta, unsigned int threshold):
    cdef int max_dist
    cdef int i,j,x,y,t,r
    cdef long[:] xi,yi
    cdef double[:] thetas, rhos, cosine, sine
    cdef unsigned int[:,:] accumulator
    cdef object acc_votes = defaultdict(list)
    max_dist = np.sqrt(np.power(width,2)+np.power(height,2))
    rhos = np.linspace(0,max_dist,max_dist/rho)
    thetas = np.deg2rad(np.arange(0,180,theta))
    accumulator = np.zeros((rhos.shape[0],thetas.shape[0]),dtype=np.uint32)
    
    cosine = np.cos(thetas)
    sine = np.sin(thetas)
    
    xi,yi = np.nonzero(image)
    
    for i in range(len(xi)):
        x = xi[i]
        y = yi[i]
        
        for t in range(len(thetas)):
            r = round(x*cosine[t] + y*sine[t])
            accumulator[r,t] +=1
            acc_votes[(r,t)].append((x,y))

    for i,j in list(acc_votes):
        if accumulator[i,j] <= threshold:
            del acc_votes[(i,j)]
    return accumulator, thetas, rhos, acc_votes