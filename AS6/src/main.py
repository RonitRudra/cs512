#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:17:44 2017

@author: ronitrudra
"""
import numpy as np
import sys
import cv2

color = np.random.randint(0,255,(10000,3))
f = open("../doc/help.txt")
help_file = f.read()
f.close()

def calcReliability(system_matrix):
    rel = []
    for i,x in enumerate(system_matrix):
        if np.all(x==0):
            rel.append(0)
        else:
            u,d,v = np.linalg.svd(x)
            r = d[1]/d[0]
            rel.append(r)
    return rel

def scaleIntensity(pt,rel):
    pt_hsv = cv2.cvtColor(np.uint8([[[pt[0],pt[1],pt[2]]]]),cv2.COLOR_BGR2HSV)
    pt_hsv[0][0][2] = pt_hsv[0][0][2]*rel
    pt = cv2.cvtColor(pt_hsv,cv2.COLOR_HSV2BGR)
    return pt
        

def drawVectors(frame,pts_old,pts_new,reliability):
    assert len(pts_new)==len(reliability),"oops"
    for i,(old,new) in enumerate(zip(pts_old,pts_new)):
        if old ==0 or new == 0:
            continue
        c = scaleIntensity(color[i],reliability[i])
        cv2.line(frame,(int(new[0]),int(new[1])),(int(old[0]),int(old[1])),c.tolist()[0][0],5)
    return frame


def lucasKanade(frame_new,frame_old,feature_points,side,ksize):
    old = cv2.cvtColor(frame_old,cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(frame_new,cv2.COLOR_BGR2GRAY)
    w,h = old.shape
    pts = []
    system_matrix = []
    
    a,b = np.gradient(old)
    c,d = np.gradient(new)
    old = cv2.GaussianBlur(old,(5,5),0)
    new = cv2.GaussianBlur(new,(5,5),0)
    Ix_ = (c-a)/2
    Iy_ = (d-b)/2
    It_ = new - old
    
    for x,y in feature_points:
        x,y = int(x),int(y)
        if x+side>w or x-side<0 or y+side>h or y-side<0:
            pts.append((0,0))
            system_matrix.append(np.zeros((2,2)))
            continue
        Ix = Ix_[(x-side-1):(x+side),(y-side-1):(y+side)]
        Iy = Iy_[(x-side-1):(x+side),(y-side-1):(y+side)]
        It = It_[(x-side-1):(x+side),(y-side-1):(y+side)]
        a11 = np.sum(np.power(Ix.ravel(),2))
        a22 = np.sum(np.power(Iy.ravel(),2))
        a_off = np.sum(Ix.ravel()*Iy.ravel())
        A = np.array([[a11,a_off],[a_off,a22]])
        b11 = -(np.sum(Ix.ravel()*It.ravel()))
        b21 = -(np.sum(Iy.ravel()*It.ravel()))
        b = np.array([[b11],[b21]])
        Ainv = np.linalg.pinv(A)
        x = np.matmul(Ainv,b)
        pts.append((x[0,0],x[1,0]))
        system_matrix.append(A)
    
    return pts,system_matrix


def main():
    if len(sys.argv)==1:
        filename = 0
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        raise ValueError("Incorrect Number of Arguments")
    flag = True

    try:
        capture = cv2.VideoCapture(filename)
    except Exception as error:
        print(error)
        
    cv2.namedWindow("Capture Window",cv2.WINDOW_AUTOSIZE)
    
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    
    ret, frame_old = capture.read()
    if(flag) :
        gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
        pts_old = cv2.goodFeaturesToTrack(gray_old, mask = None, **feature_params)
        flag = False

    mask = np.zeros_like(frame_old)
    
    while True:
        ret, frame_new = capture.read()
        if ret:
            pts_old = [(x,y) for x,y in pts_old[:,0,:]]
            pts_new,A = lucasKanade(frame_new,frame_old,pts_old,7,3)
            pts_new = [(p1[0]+p2[0],p1[1]+p2[1]) for p1,p2 in zip(pts_old,pts_new)]
            reliability = calcReliability(A)
            mask = drawVectors(mask,pts_old,pts_new,reliability)
            img = cv2.add(frame_new,mask)
            cv2.imshow("Capture Window",img)
            k = cv2.waitKey(10)
            if k == ord('e'):
                cv2.destroyAllWindows()
                break
            if k == ord('h'):
                print(help_file)
            
            if k == ord('r'):
                mask = np.zeros_like(frame_old)

            if k == ord('p'):
                l = cv2.waitKey(0)
                if l == ord('p'):
                    pass
                if l == ord('e'):
                    cv2.destroyAllWindows()
                    break
        
            frame_old = frame_new.copy()
            gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
            pts_old = cv2.goodFeaturesToTrack(gray_old, mask = None, **feature_params)
  
if __name__ == "__main__":
    main()
        
    
    