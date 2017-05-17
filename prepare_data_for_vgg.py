from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.models import load_model

import cv2
#from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import tqdm
import scipy

size = 56,56
rows, columns = 56, 56
n_images = 12500
num_classes = 4

#function to read in and resize a color image
def read_colimage(src):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (size), interpolation=cv2.INTER_NEAREST)
    return im

def read_data():
    #read in images
    print('reading images...')
    imList = np.ndarray((n_images, rows, columns, 3), dtype=np.uint8)
    x = 0
    ct = 0
    for infile in glob.glob('C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/screencaps_fighting/game_*/*.jpg'):
        imList[ct] = read_colimage(infile)
        ct += 1
        if infile[-7:] == '500.jpg':
            x += 1
            print(x)
    print('done reading images')
    
    #read in labels
    print('reading labels...')
    y = []
    for infile in glob.glob('C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/screencaps_fighting/game_*/*.csv'):
        newY = pd.read_csv(infile, header=None, names=['stroke'])
        newY = newY['stroke'].values.tolist()
        y.extend(newY)
    y = np.array(y)
    print('done reading labels')
    
    return imList, y

#x_all, y_all = read_data()

#y_all2 = keras.utils.to_categorical(y_all, num_classes)
X_train = x_all[:10000]
y_train = y_all[:10000]
X_test = x_all[10000:]
y_test = y_all[10000:]


#%% Create demo data
def makedata(basepath):
    if os.path.exists(basepath): return
    #from keras.datasets import cifar10
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #if cfg.randomlbls: y_train = np.random.randint(0, len(np.unique(y_train)), y_train.shape)
    obj_classes = ['nothing', 'left', 'right', 'punch'] 
    for (X_data, y_data, bp) in [(X_train, y_train, trainfolder), (X_test, y_test, testfolder)]:
        for c in obj_classes: os.makedirs(os.path.join(bp, c), exist_ok=True)
        for i, (im, lbl) in tqdm.tqdm(enumerate(zip(X_data, y_data)), desc='Making data folder', total=len(y_data)):
            pn = os.path.join(bp, obj_classes[int(lbl)], "%d.png" % i)
            if not os.path.exists(pn): scipy.misc.imsave(pn, scipy.misc.imresize(im, (56,) * 2, interp='bicubic'))


myPath = 'C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/vgg_input'
trainfolder = os.path.join(myPath, 'train')
testfolder = os.path.join(myPath, 'test')

makedata(myPath) # Comment out this line to use your own data

        
        
