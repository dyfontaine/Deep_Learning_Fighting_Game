# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:54:40 2017

@author: Dylan
"""

import keras  
from PIL import ImageGrab
import numpy as np
import cv2
import win32api
import win32ui
import time
import os
import matplotlib.pyplot as plt

#xModel = load_model('C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/model1_fighting.h5')

myButtons = {
        'left': 0x41,
        'right': 0x44,
        'hit': 0x4A
        }

sleep_time=0.1

#propINdata = [0.18433,0.0472,0.0498667,0.161367,0.2223,0.079533,0.0924,0.079433,0.083567]

pressList = []
#%%

game = 'Rumblah: Flash Fighting Engine - Free online games at Agame.com - Google Chrome'
Wind = win32ui.FindWindow(None,game)
Wind.SetFocus()
#press and release alt key to ensure SetForegro.. works
win32api.keybd_event(0x12,0,)
win32api.keybd_event(0x12,0,2)
Wind.SetForegroundWindow()

#%%
for i in range(0,100):
    #take screenshot and crop it
    img = ImageGrab.grab(bbox=(0,0,1366,768))
    img_np = np.array(img)
    img_cropped = img_np[80:730,194:1170]
    downScaled = cv2.resize(img_cropped, (56,56), interpolation=cv2.INTER_NEAREST)
    
    # alter weights of red/green
    #for i in range(0,56):
    #    for j in range(0,56):
    #        downScaled[i,j,0] = downScaled[i,j,0]*0
    
    AddedDim = np.expand_dims(downScaled, axis=0)
    xNEW = AddedDim.reshape(1, 56*56*3)
    xNEW= xNEW.astype('float32')
    xNEW /= 255
    

    
    #xpClass = xModel.predict_classes(xNEW, batch_size=1, verbose=0)
    #xpClass = xpClass[0]
    ########
    xprobs = xModel.predict_proba(xNEW, batch_size=1, verbose=0)
    
    ################################################
    #make do nothing class ZERO
    #prob_nothing = xprobs[0,0]
    xprobs[0,0] = 0
    #first = True
    #for j in range(0,xprobs.shape[1]):
    #jjjjjj    if first == True:
    #        first = False
    #    else:
    #        xprobs[0,j] += prob_nothing/8
    #reduce DOWN probability
    #xprobs[0,4] = 0.75*xprobs[0,4]
    ################################################
    
    #_______ weighting inversely by occurences in train data
    #for w in range(0,xprobs.shape[1]):
    #    xprobs[0,w] = xprobs[0,w] /propINdata[w]
    
    #__________________________


    my_max = 0
    kk = 0
    max_index = -99
    for w in range(0,xprobs.shape[1]):
        if xprobs[0,w] > my_max:
            my_max = xprobs[0,w]
            max_index = w
        kk += 1
    xpClass = max_index
    ########    
    pressList.append(xpClass)
    if (xpClass == 1):
        win32api.keybd_event(myButtons['left'],0,)
        time.sleep(sleep_time)
        win32api.keybd_event(myButtons['left'],0,2)
    elif (xpClass==2):
        win32api.keybd_event(myButtons['right'],0,)
        time.sleep(sleep_time)
        win32api.keybd_event(myButtons['right'],0,2)
    elif (xpClass==3):
        win32api.keybd_event(myButtons['hit'],0,)
        time.sleep(sleep_time)
        win32api.keybd_event(myButtons['hit'],0,2)
    else:
        time.sleep(sleep_time)
    time.sleep(0.1)

        

#0 none
#1 left
#2 right
#3 up
#4 down
#5 up-left
#6 down-left
#7 up-right
#8 down-right
#plt.imshow(downScaled, cmap = 'gray')

#return to Spyder
Wind2 = win32ui.FindWindow(None,'Spyder (Python 3.5)')
Wind2.SetFocus()
win32api.keybd_event(0x12,0,)
win32api.keybd_event(0x12,0,2)
Wind2.SetForegroundWindow()
