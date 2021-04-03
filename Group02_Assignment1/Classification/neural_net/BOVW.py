#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:20:24 2021

@author: harshit
"""
import numpy as np
import cv2
Thresh = 1000000
import random
import os
i1 = 1
sizeofBOVW = 32
Epochs = 100
def build_hist(data):
    arr = [0]*8
    for i in data:
        arr[i//32] += 1
    return arr

def Build_hist_for_patch(img_data,row,column):
    red = []
    green = []
    blue = []
    for i in range(row,row+32):
        for j in range(column,column+32):
            red.append(img_data[i][j][0])
            green.append(img_data[i][j][1])
            blue.append(img_data[i][j][2])
    hist = build_hist(red)+build_hist(green)+build_hist(blue)
    return np.array(hist)


def img_feature_extraction(img_data):
    img = []
    for row in range(0,len(img_data),32):
        for col in range(0,len(img_data[0]),32):
            img.append(Build_hist_for_patch(img_data,row,col))
    return img

def BOVW(data):
    BOVWDATA = [np.zeros(32) for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            BOVWDATA[i][data[i][j]] += 1
        BOVWDATA[i] /= len(data[i])
    return BOVWDATA

def ExtractFeatureVector(img_data,clusterPoints):
    img_data = cv2.copyMakeBorder(img_data, 0,(32-len(img_data)%32)%32 , 0, (32-len(img_data[0])%32)%32, cv2.BORDER_REFLECT)
    img_data = img_feature_extraction(img_data)
    FeatureVector = np.zeros(32)
    for i in range(len(img_data)):
        FeatureVector[np.argmin([sum((img_data[i]-clusterPoints[k])**2) for k in range(len(clusterPoints))])] += 1
    FeatureVector /= len(img_data)
    return FeatureVector

def KmeansClustering(data,prevclasses,clusterPoints):
    classes = [[-1]*len(data[i]) for i in range(len(data))]
    updateCount = 0
    DistMeas = 0
    SumNewClusterPoints = [np.zeros(24) for i in range(32)]
    CountNewClusterPoints = [0]*32
    for i in range(len(data)):
        for j in range(len(data[i])):
            classes[i][j] = np.argmin([sum((data[i][j]-clusterPoints[k])**2) for k in range(len(clusterPoints))])
            if classes[i][j] != prevclasses[i][j]:
                updateCount += 1
            DistMeas += sum((data[i][j] - clusterPoints[classes[i][j]])**2)
            CountNewClusterPoints[classes[i][j]] += 1
            SumNewClusterPoints[classes[i][j]] += data[i][j]
    NewClusterPoints = [SumNewClusterPoints[i]/CountNewClusterPoints[i] if CountNewClusterPoints[i]!=0 else np.zeros(24) for i in range(32)]
    print(DistMeas)
    global Epochs
    Epochs -= 1
    if updateCount == 0 or Epochs == 0:
        return (NewClusterPoints,BOVW(classes))
    else:
        return KmeansClustering(data,classes,NewClusterPoints)
    
p_dir = "/home/harshit/Sem6/DL/Group02/Classification/Image_Group02/train"
coast = os.path.join(p_dir,"coast")
industrial_area = os.path.join(p_dir,"industrial_area")
pagoda = os.path.join(p_dir,"pagoda")
data = []
TrainClasses = []
for i in os.listdir(coast):
    img_data = cv2.imread(os.path.join(coast,i))
    img_data = cv2.copyMakeBorder(img_data, 0,(32-len(img_data)%32)%32 , 0, (32-len(img_data[0])%32)%32, cv2.BORDER_REFLECT)
    img_data = img_feature_extraction(img_data)
    data.append(img_data)
    TrainClasses.append(0)
for i in os.listdir(industrial_area):
    img_data = cv2.imread(os.path.join(industrial_area,i))
    img_data = cv2.copyMakeBorder(img_data, 0,(32-len(img_data)%32)%32 , 0, (32-len(img_data[0])%32)%32, cv2.BORDER_REFLECT)
    img_data = img_feature_extraction(img_data)
    data.append(img_data)
    TrainClasses.append(1)
for i in os.listdir(pagoda):
    img_data = cv2.imread(os.path.join(pagoda,i))
    img_data = cv2.copyMakeBorder(img_data, 0,(32-len(img_data)%32)%32 , 0, (32-len(img_data[0])%32)%32, cv2.BORDER_REFLECT)
    img_data = img_feature_extraction(img_data)
    data.append(img_data)
    TrainClasses.append(2)
    

classes = [[-1]*len(data[i]) for i in range(len(data))]
ClusterPoints = []
for i in range(32):
    p = random.randint(0,len(data)-1)
    ClusterPoints.append(data[p][random.randint(0,len(data[p])-1)])
BOVW_DATA = KmeansClustering(data,classes,ClusterPoints)
ClusterPoints = BOVW_DATA[0]
Image_Data = BOVW_DATA[1]
