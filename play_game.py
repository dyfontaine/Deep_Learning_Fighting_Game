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

sleep_time=0.1 #amount of time to hold down a button for

pressList = [] #list that keeps track of keys that are pressed by AI


#%%

#open game window
game = 'Rumblah: Flash Fighting Engine - Free online games at Agame.com - Google Chrome'
Wind = win32ui.FindWindow(None,game)
Wind.SetFocus()

#press and release alt key to ensure SetForegroundWindow works
win32api.keybd_event(0x12,0,)
win32api.keybd_event(0x12,0,2)
Wind.SetForegroundWindow()

#%%

#loop to play game for 200 screenshots 
for i in range(0,200):
    #take screenshot and crop it
    img = ImageGrab.grab(bbox=(0,0,1366,768))
    img_np = np.array(img)
    img_cropped = img_np[80:730,194:1170]
    downScaled = cv2.resize(img_cropped, (56,56), interpolation=cv2.INTER_NEAREST)
    
    #add dimension so we can feed it in to model
    AddedDim = np.expand_dims(downScaled, axis=0)
    xNEW = AddedDim.reshape(1, 56*56*3)
    xNEW= xNEW.astype('float32')
    xNEW /= 255
    
    #make do nothing class ZERO
    xprobs = xModel.predict_proba(xNEW, batch_size=1, verbose=0)
    xprobs[0,0] = 0
          
    #weighting inversely by occurences in train data
    #for w in range(0,xprobs.shape[1]):
    #    xprobs[0,w] = xprobs[0,w] /propINdata[w]
    
    #get class with highest probability
    my_max = 0
    kk = 0
    max_index = -99
    for w in range(0,xprobs.shape[1]):
        if xprobs[0,w] > my_max:
            my_max = xprobs[0,w]
            max_index = w
        kk += 1
    xpClass = max_index
    
    #list that records which key is being pressed by AI
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
      
#plt.imshow(downScaled, cmap = 'gray')

#return to Spyder
Wind2 = win32ui.FindWindow(None,'Spyder (Python 3.5)')
Wind2.SetFocus()
win32api.keybd_event(0x12,0,)
win32api.keybd_event(0x12,0,2)
Wind2.SetForegroundWindow()
