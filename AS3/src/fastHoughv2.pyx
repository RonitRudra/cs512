#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:24:43 2017

@author: ron
"""
import numpy as np
from collections import defaultdict
cimport numpy as np
from libc.math cimport pow,sqrt,round,abs
cimport cython

def fastHoughv2(np.ndarray image, int width, int height, int rho_res, int theta_res, unsigned int threshold):
    cdef long[:] xi,yi
    cdef int max_dist
    cdef int i,j,x,y,t,dx,dy,rho_id, rho_len, rho_min,rho_dist,ri
    cdef double r
    cdef double[:] thetas, rhos, cosine, sine
    cdef unsigned int[:,:] accumulator
    cdef object acc_votes = defaultdict(list)
    xi,yi = np.nonzero(image)
    max_dist = int(round(sqrt(pow(width,2)+pow(height,2))))
    rhos = np.linspace(-max_dist,max_dist,int(2*max_dist/rho_res)+1)
    rho_len = len(rhos)
    thetas = np.deg2rad(np.arange(0,180,theta_res))
    accumulator = np.zeros((rhos.shape[0],thetas.shape[0]),dtype=np.uint32)
    
    cosine = np.cos(thetas)
    sine = np.sin(thetas)
    
    for i in range(len(xi)):
        x = xi[i]
        y = yi[i]
        
        for t in range(len(thetas)):
            r = x*cosine[t] + y*sine[t]
#           rho_id = np.nonzero(np.abs(rhos-r)==np.min(np.abs(rhos-r)))[0][0]
            rho_min = 10000
            for ri in range(rho_len):
                rho_dist = int(round((abs(rhos[ri]-r))))
                if rho_dist < rho_min:
                    rho_min = rho_dist
                    rho_id = ri
                    
                
            accumulator[rho_id,t] +=1
            acc_votes[(rho_id,t)].append((x,y))

    for i,j in list(acc_votes):
        if accumulator[i,j] <= threshold:
            del acc_votes[(i,j)]
    return accumulator, thetas, rhos, acc_votes