#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:17:38 2017

@author: ron
"""

import sys

from helper import Helper
from image_class import Image

flag = False
h = Helper()
filename = None

if len(sys.argv)<2:
    flag=True
elif len(sys.argv)==2:
    if sys.argv[1].endswith('.jpg') or sys.argv[1].endswith('.png'):
        filename = sys.argv[1]
    elif sys.argv[1] =="-x":
        h.get_filename()
        filename = h.filename
        if len(filename)>1:
            filename = h.select_file()
        else:
            filename = h.filename
    else:
        print("Bad Argument. Exiting...")
        exit()
                
functions = {'i':'reload',
                 'w':'save_image',
                 'g':'grayscale',
                 'G':'myGrayscale',
                 'c':'cycleColorChannel',
                 's':'grayScaleAndSmooth',
                 'S':'mygrayScaleAndSmooth',
                 'd':'downSampleNoSmoothing',
                 'D':'downSample',
                 'x':'conv_GS_DerivativeX',
                 'y':'conv_GS_DerivativeY',
                 'm':'gradient_norm',
                 'n':'plotGradientVectors',
                 'r':'rotate',
                 'h':'showHelp',}
image = Image(h.curr_path,functions,filename,flag)
if image.vc_flag == False:
    while (True):
        h.showMenu()
        h.interfaceAction(image)
        image.reload()
        print("Press any key to return to the menu or \'e\' to exit")
        keyPress = input()
        if keyPress == 'e':
            break
elif image.vc_flag == True:
    while(True):
        h.showMenu()
        keyPress = h.interfaceAction(image)
        while(True):
            image.cameraProc(keyPress)



        

