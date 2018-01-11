#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:32:10 2017

@author: ron
"""
import numpy as np
from collections import defaultdict

def houghTransform(image, width, height, rho, theta,threshold):
    max_dist = np.sqrt(np.power(width,2)+np.power(height,2))
    rhos = np.linspace(0,max_dist,max_dist/rho)
    thetas = np.deg2rad(np.arange(0,180,theta))
    accumulator = np.zeros((rhos.shape[0],thetas.shape[0]),dtype=np.uint8)
    
    cosine = np.cos(thetas)
    sine = np.sin(thetas)
    
    xi,yi = np.nonzero(image)
    acc_votes = defaultdict(list)
    
    for i in range(len(xi)):
        x = xi[i]
        y = yi[i]
        
        for t in range(len(thetas)):
            r = int(x*cosine[t] + y*sine[t])
            accumulator[int(round(r)),t] +=1
            acc_votes[(int(round(r)),t)].append((x,y))

    for i,j in list(acc_votes):
        if accumulator[i,j] <= threshold:
            del acc_votes[(i,j)]
    return accumulator, thetas, rhos, acc_votes