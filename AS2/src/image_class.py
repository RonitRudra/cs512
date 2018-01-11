#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:33:42 2017

@author: ron
"""

import cv2
import numpy as np
#import sys
import os
from itertools import cycle
import matplotlib.pyplot as plt

class Image():
    
    def __init__(self,img_path,functions,filename=None,vc_flag=False):
        self.vc_flag = vc_flag
        self.functions = functions
        if self.vc_flag == False:
            self.img_path = os.path.join(img_path,"../data/",filename)
            self.img = cv2.imread(self.img_path)
        else:
            self.img_capture = cv2.VideoCapture(0)
            self.img = self.cameraProc()
        
        
    def function_call(self,fkey,*args,**kwargs):
        func_name = self.functions.get(fkey)
        if func_name and hasattr(self, func_name):
            return getattr(self, func_name)(*args, **kwargs)
        else:
            return None
        
    def reload(self):
        self.img = cv2.imread(self.img_path)
        
    def save_image(self):
        cv2.imwrite("out.jpg",self.img)
        
    def grayscale(self,display=True):
        ## Only display image if flag is True
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if display==True:
            self.showImage()
    
    def myGrayscale(self,display=True):
        self.img = 0.229*self.img[:,:,2] + 0.587*self.img[:,:,1] + 0.114*self.img[:,:,0]
        if display==True:
            self.showImage()
        
    def cycleColorChannel(self):
        # Make list of RGB channels
        channel_list = [self.img[:,:,0],self.img[:,:,1],self.img[:,:,2]]
        # Create a cycle iterator
        cycles = cycle(channel_list)
        # indefinitely cycle through channel till exit key pressed
        for c in cycles:
            cv2.imshow("Image",c)
            k = cv2.waitKey(0)
            if k==ord('e'):
                cv2.destroyAllWindows()
                break
    
    def grayScaleAndSmooth(self):
        if len(self.img.shape)==3:
            self.grayscale(False)
        cv2.imshow("Image",self.img)
        # lambda expression used to pass additional parameter to slider
        # this ensures I don't have to make different slider definitions for rotation and gradient vectors
        cv2.createTrackbar('Kernel Size',"Image",3,255,lambda x: self._slider(x, "SMOOTH"))
        k = cv2.waitKey()
        if k == ord('e'):
            cv2.destroyAllWindows()

    def mygrayScaleAndSmooth(self):
        # Manual convolution is in slider function to reduce number of unnecessary functions
        if len(self.img.shape)==3:
            self.grayscale(False)
        cv2.imshow("Image",self.img)
        cv2.createTrackbar('Kernel Size',"Image",3,255,lambda x: self._slider(x, "MYSMOOTH"))
        k = cv2.waitKey()
        if k == ord('e'):
            cv2.destroyAllWindows()
        
    def downSampleNoSmoothing(self):
        # Simply remove alternate rows and columns
        self.img = self.img[0::2,0::2,:]
        self.showImage()
                
    def downSample(self):
        # cv2.pyrDown does downsampling by 2 in both axis and applies a 5x5 gaussian filter
        self.img = cv2.pyrDown(self.img)
        self.showImage()
        
    def conv_GS_DerivativeX(self):
        ## Applies sobel operator in x direction
        ## For purposes of assignment, there are only two combinations hardcoded in main.py
        ## 1,0 and 0,1 i.e either an x derivative or a y derivative
        # Check if image is already in grayscale
        if len(self.img.shape)==3:
            self.grayscale(False)
        self.img = cv2.Sobel(self.img,cv2.CV_32F,1,0,ksize=5)
        # Normalize
        # can use np.max to find find maximum value of pixel
        # Dividing by maximum will normalize all values between 0 and 1
        self.img = self.img/(np.max(self.img))
        self.showImage()
        
    def conv_GS_DerivativeY(self):
        ## Applies sobel operator in y direction
        if len(self.img.shape)==3:
            self.grayscale(False)
        self.img = cv2.Sobel(self.img,cv2.CV_32F,0,1,ksize=5)
        self.img = self.img/(np.max(self.img))
        self.showImage()
        
        
    def gradient_norm(self):
        if len(self.img.shape) == 3:
            self.grayscale(False)
        dx, dy = np.gradient(self.img)
        grads = np.sqrt(dx**2 + dy**2)
        grads /=np.max(grads)
        self.showImage("Gradients",grads)
        
    
    def plotGradientVectors(self):
        if len(self.img.shape) == 3:
            self.grayscale(False)
        dx,dy = np.gradient(self.img)
        cv2.imshow("Image",self.img)
        cv2.createTrackbar('Vector Density',"Image",50,255,lambda x: self._slider(x, "VECTOR",dx,dy))
        k = cv2.waitKey()
        if k == ord('e'):
            cv2.destroyAllWindows()
        
    def rotate(self):
        cv2.imshow("Image",self.img)
        cv2.createTrackbar('Vector Density',"Image",50,255,lambda x: self._slider(x, "ROTATE"))
        k = cv2.waitKey()
        if k == ord('e'):
            cv2.destroyAllWindows()
    
    def showImage(self,window = "Image",image = None):
        if image is None:
            image = self.img
        cv2.imshow("Image",image)
        if self.vc_flag==True:
            waitTime = 30
        else:
            waitTime = 0
        k = cv2.waitKey(waitTime)
        if k == ord('e'):
            cv2.destroyAllWindows()
            
    def _slider(self,n,flag,dx=None,dy=None):
        # Do not call manually
        # To be only called through cv2.createTrackbar callback
        if flag == "SMOOTH":
            kernel = np.ones((n,n),np.float32)/(n*n)
            res = cv2.filter2D(self.img,-1,kernel)
            cv2.imshow("Image",res)
            
        elif flag == "MYSMOOTH":
            # manual slider for smoothing
            # Only do smoothing if n is at least 3
            if n >= 3:
                kernel = np.ones((n,n),np.float32)/(n*n)
                # size of zero padding
                # kernel size can be input as even so round to nearest whole integer
                P = int(round((n-1)/2))
                img_pad = np.pad(self.img,(P,P), 'constant')
                res = np.zeros_like(self.img)
                # Naive implementation of 2D convolution
                for i in range(res.shape[0]):
                    for j in range(res.shape[1]):
                        res[i,j] = np.sum(img_pad[i:(i+n),j:(j+n)]*kernel)
            cv2.imshow("Image",res)
        
        elif flag == "ROTATE":
            #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
            theta = np.interp(n,[0,255],[0,360])
            (h, w) = self.img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -theta, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            res = cv2.warpAffine(self.img, M, (nW, nH),flags=cv2.WARP_INVERSE_MAP)
            cv2.imshow("Image",res)
        
        elif flag == "VECTOR":
            plt.quiver(dx[0::n,0::n],dy[0::n,0::n])
            plt.savefig("../data/vector.jpg",dpi=80)
            self.img = cv2.imread("../data/vector.jpg")
            cv2.imshow("Image",self.img)
        else:
            pass
        
    def cameraProc(self,fkey=None):
            valid_frame, image = self.img_capture.read()
            self.img = image
            if fkey is not None:
                self.function_call(fkey)
                
    
    def showHelp(self):
        f = open("../doc/help.txt",'r')
        help_content = f.read()
        f.close()
        print(help_content)
    
    