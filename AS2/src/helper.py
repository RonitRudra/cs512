#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:26:45 2017

@author: ron
"""

#import sys
import os

class Helper():
    '''
    Contains functions for traversing directory and fetching image filenames.
    Also contains functions for menu operations and image_class function calls.
    '''
    
    
    def __init__(self):
        # save path of program. Makes it system independent
        # This is useful as there are multiple .py files and data files in other directories
        self.curr_path = os.path.abspath(path = "")
        self.filename = []


    def get_filename(self):
        for file in os.listdir(os.path.join(self.curr_path,"../data/")):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.filename.append(file)
    
    def select_file(self):
        print("More than one file present in the directory. Please choose number from below:")
        for i,file in zip(range(len(self.filename)),self.filename):
            print("%d. %s"%(i+1,file))
        index = int(input())
        if index not in range(1,len(self.filename)):
            print("Invalid Input..Please Choose Again!!")
            self.select_file()
        return self.filename[index-1]
    
    def showMenu(self):
        print("\n")
        print("What would you like to do?\nSelect an option by pressing the corresponding key\nand pressing Return.")
        print("i - Reload Original Image")
        print("w - Save the Current Image")
        print("g - Convert Image to Grayscale (Builtin)")
        print("G - Convert Image to Grayscale (Custom)")
        print("c - Cycle Color Channels")
        print("s - Convert to Grayscale and Smooth (Builtin)")
        print("S - Convert to Grayscale and Smooth (Custom)")
        print("d - Downsample by 2 without Smoothing")
        print("D - Downsample by 2 with Smoothing")
        print("x - Get X Derivative")
        print("y - Get Y Derivative")
        print("m - Get Normalized Gradients")
        print("n - Plot Gradient Vectors")
        print("r - Rotate")
        print("h - Show Help")
        print("\n")
        
    def interfaceAction(self,image_class):
        keyPress = input()
        # Check if input is a valid dictionary key
        if keyPress in image_class.functions.keys():
            if image_class.vc_flag == False:
                image_class.function_call(keyPress)
            elif image_class.vc_flag == True:
                return keyPress
        else:
            print("Invalid Choice")
            return None
        
                    

    