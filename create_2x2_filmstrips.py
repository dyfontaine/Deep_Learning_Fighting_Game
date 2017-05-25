#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:06:41 2017

@author: adidier
"""

from __future__ import print_function
import os
import numpy as np
import cv2
import glob
import re

size = 56,56
rows, columns = 56, 56
n_images = 20000

imList = np.ndarray((n_images, rows, columns, 3), dtype=np.uint8)
label = ["na"] * n_images


base_path = '/home/lab.analytics.northwestern.edu/adidier/Documents/Deep_Learning_Project/vgg_input/train/'
directories = [base_path + 'left', base_path + 'right', base_path + 'punch', base_path + 'nothing']
label_names = {base_path + 'left': 'left' , base_path + 'right': 'right', base_path + 'punch': 'punch', base_path + 'nothing': 'nothing'}

#load all the images
for directory_in_str in directories:
     directory = os.fsencode(directory_in_str)
     for file in os.listdir(directory):
        filename = os.fsdecode(file)   
        index = int(re.findall(r'\d+', filename)[0])
        path = directory_in_str + '/' + filename
        imList[index] = cv2.imread(path, cv2.IMREAD_COLOR)
        label[index] = label_names[directory_in_str]

indices = []
for directory_in_str in directories:
     directory = os.fsencode(directory_in_str)
     for file in os.listdir(directory):
        filename = os.fsdecode(file)   
        index = int(re.findall(r'\d+', filename)[0])
        indices.append(index)


#save images to a 2x2 grid
frows, fcolumns = 56*2, 56*2
film_strip_im = np.ndarray((n_images-4, frows, fcolumns, 3), dtype = np.uint8)
for i in range(3,n_images-1):  
    row1 = np.concatenate((imList[[i-3]][0],imList[[i-2]][0]), axis = 1)
    row2 = np.concatenate((imList[[i-1]][0],imList[[i]][0]), axis = 1)
    film_strip_im[i-3] = np.concatenate((row1,row2),axis=0)    


#start at index 3 because the first image contains images 0-3 in the original folder,
#the label corresponds to the last image at index 3 in labels
index = 3
film_base_path = '/home/lab.analytics.northwestern.edu/adidier/Documents/Deep_Learning_Project/film_vgg_input/train/'
for image in film_strip_im:
    film_label = label[index]
    path = film_base_path + film_label + '/' + str(index) + '.png'
    cv2.imwrite(path, image)
    index += 1
    
################ test images ############################
size = 56,56
rows, columns = 56, 56
n_images = 5000

imList_test = np.ndarray((n_images, rows, columns, 3), dtype=np.uint8)
label_test = ["na"] * n_images


base_path = '/home/lab.analytics.northwestern.edu/adidier/Documents/Deep_Learning_Project/vgg_input/test/'
directories = [base_path + 'left', base_path + 'right', base_path + 'punch', base_path + 'nothing']
label_names = {base_path + 'left': 'left' , base_path + 'right': 'right', base_path + 'punch': 'punch', base_path + 'nothing': 'nothing'}

#load all the images
for directory_in_str in directories:
     directory = os.fsencode(directory_in_str)
     for file in os.listdir(directory):
        filename = os.fsdecode(file)   
        index = int(re.findall(r'\d+', filename)[0])
        path = directory_in_str + '/' + filename
        imList_test[index] = cv2.imread(path, cv2.IMREAD_COLOR)
        label[index] = label_names[directory_in_str]

indices = []
for directory_in_str in directories:
     directory = os.fsencode(directory_in_str)
     for file in os.listdir(directory):
        filename = os.fsdecode(file)   
        index = int(re.findall(r'\d+', filename)[0])
        indices.append(index)


#save images to a 2x2 grid
frows, fcolumns = 56*2, 56*2
film_strip_im_test = np.ndarray((n_images-4, frows, fcolumns, 3), dtype = np.uint8)
for i in range(3,n_images-1):  
    row1 = np.concatenate((imList_test[[i-3]][0],imList_test[[i-2]][0]), axis = 1)
    row2 = np.concatenate((imList_test[[i-1]][0],imList_test[[i]][0]), axis = 1)
    film_strip_im_test[i-3] = np.concatenate((row1,row2),axis=0)    


#start at index 3 because the first image contains images 0-3 in the original folder,
#the label corresponds to the last image at index 3 in labels
index = 3
film_base_path = '/home/lab.analytics.northwestern.edu/adidier/Documents/Deep_Learning_Project/film_vgg_input/test/'
for image in film_strip_im:
    film_label = label[index]
    path = film_base_path + film_label + '/' + str(index) + '.png'
    cv2.imwrite(path, image)
    index += 1    