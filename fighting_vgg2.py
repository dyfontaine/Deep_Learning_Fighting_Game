#%% read data in with original dimensions
from __future__ import print_function

import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization,Flatten, normalization
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.models import load_model

import cv2
#from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob

size = 128,128
rows, columns = 128,128
n_images = 12500
input_shape = (rows, columns, 3)

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
    for infile in glob.glob('C:/Users/mwcho/Desktop/trainimages/screencaps_fighting/game_*/*.jpg'):
#    for infile in glob.glob('C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/screencaps_fighting/game_*/*.jpg'):
        imList[ct] = read_colimage(infile)
        ct += 1
        if infile[-7:] == '500.jpg':
            x += 1
            print(x)
    print('done reading images')
    
    #read in labels
    print('reading labels...')
    y = []
    for infile in glob.glob('C:/Users/mwcho/Desktop/trainimages/screencaps_fighting/game_*/*.csv'):
#    for infile in glob.glob('C:/Users/Dylan/Documents/Northwestern/Spring 2017/Deep Learning/Deep_Learning_Project_LOCAL_FILES/screencaps_fighting/game_*/*.csv'):
        newY = pd.read_csv(infile, header=None, names=['stroke'])
        newY = newY['stroke'].values.tolist()
        y.extend(newY)
    y = np.array(y)
    print('done reading labels')
    
    return imList, y

#%% define training and test
#MAIN
batch_size = 128
num_classes = 4
epochs = 50

x_all, y_all = read_data()

#fix values in y
#left 9 : 1
#right 10 : 2
#nothing  : 0
#hit   11 : 3

for yyy in range(0,y_all.shape[0]):
    if y_all[yyy] == 9:
        y_all[yyy] = 1
    elif y_all[yyy] == 10:
        y_all[yyy] = 2
    elif y_all[yyy] == 11:
        y_all[yyy] = 3
        

#x_all_flat = x_all.reshape(n_images, rows*columns*3)
x_train, x_test = x_all[:10000,], x_all[10000:,]
y_train, y_test = y_all[:10000,], y_all[10000:,]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#%%
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model architecture
model = Sequential()
model.add(Conv2D(96, kernel_size=(7, 7),strides=2,padding='same', activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))
model.add(Conv2D(256, kernel_size=(5, 5),strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(512, kernel_size=(3, 3),strides=1, padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3),strides=1, padding='same',activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3),strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
for i in range(epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=0,
                        # to see if you are overfitting
                        validation_data=(x_test, y_test))
    print(history.history)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Heatmaps 2
import numpy as np, matplotlib.pyplot as plt, skimage.exposure, skimage.color, skimage.filters, tqdm
def hide_axes(ax): ax.set_xticks([]), ax.set_yticks([])
class_index = {str(i): str(i) for i in range(4)}
# 4 is for the 4 keystroke classes

def predict(x):
    # our model.predict requires 1, row*column*3
    x = x.reshape(1,56*56*3)
    # probabilities of predictions[0]
    return model.predict(x)

class Heatmap:
    def __init__(self, model):
        self.nclasses    = model.output_shape[1]
        self.model       = model
    
    def make_masks(self, im, n=8, maskval=0.0):
        masks = []
        xwidth, ywidth = int(np.ceil(im.shape[1]/n)), int(np.ceil(im.shape[2]/n))
        for i in range(n):
            for j in range(n):
                # fill with 1s of shape (28,28)
                mask = np.ones(im.shape[1:3])
                mask[(i*xwidth):((i+1)*xwidth), (j*ywidth):((j+1)*ywidth)] = maskval
                #apply a gaussian blur
                mask = skimage.filters.gaussian(mask, 0.5)                
                masks.append(mask)
        return np.array(masks)
    
    def gray2rgb(self, im): return np.concatenate(3 * (im[..., np.newaxis],), axis=-1)

    def predict_masks(self, im, masks, heatmaps, topclasses):
        for m in tqdm.tqdm(masks):
            prediction = predict(im*self.gray2rgb(m))[0]
            for c in range(4):
                clsnum, clsname = topclasses[c]
                heatmaps[c] += (prediction[clsnum]*m)
                
    def explain_prediction_heatmap(self, img, title='Heatmap', nmasks=(9, 7, 5, 3, 2)):
        # plt image with title
        plt.imshow(img), plt.xticks([]), plt.yticks([]), plt.title('Pre-cropped image'), plt.show()
        # (1,28,28,3)
        im = np.expand_dims(img, axis=0)
        # apply gaussian blur at different intervals and sum along rows
        masks = np.concatenate([self.make_masks(im, n=i) for i in nmasks])
        masknorm = masks.sum(axis=0)
        
        #class probabilities
        preds = predict(im)[0]
        # give class label sorted from high to low probabilities
        predidx = np.argsort(preds)[::-1]
        topclasses = {}
        # for index and value in sorted class labels from 0 to 4
        # print the class label, the prob, and the str class label
        for i, p in enumerate(predidx[:4]):
            print(newdict[p], preds[p])
            topclasses[i] = (newdict[p])
        # top class for index is key (index) value (class prediction)    
        print('Top classes: ', topclasses)
        
        # (5,28,28)= (5,) + (28,28)
        heatmaps = np.zeros((4,) + im.shape[1:3])
        self.predict_masks(im, masks, heatmaps, topclasses)
        for h in heatmaps: h = h / masknorm
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes[0,0].imshow(img), axes[1,0].imshow(img)        
        axes[0,0].set_title(title)
        hide_axes(axes[0,0]), hide_axes(axes[1,0])       
        predictions = np.sum(heatmaps, axis=(1,2,))
        predictions /= predictions.max()
        for n, i in enumerate(np.argsort(predictions)[::-1][:4]):
            h = ((255 * heatmaps[i])/heatmaps[i].max()).astype('uint16')
            h = skimage.filters.gaussian(h, 0.5)
            h = skimage.exposure.equalize_adapthist(h)    
            h = skimage.filters.gaussian(h, 0.5)            
            axes[0, n+1].imshow(self.gray2rgb(h))
            maskim = np.squeeze(im[:, :, :, ::-1]) * self.gray2rgb(h) * (0.5 + 0.5*predictions[i])
            maskim -= maskim.min()
            maskim /= maskim.max()
            axes[1, n+1].imshow(maskim)  
            hide_axes(axes[0, n+1]), hide_axes(axes[1, n+1])        
            axes[0, n+1].set_title(topclasses[i][1] + ': %0.1f%%' % (100*predictions[i]/predictions.sum()))
        fig.tight_layout()
        plt.show()
        return heatmaps

# you must apply heatmap to existing model, while data remains separate
# data is not contained in the model
heatmap = Heatmap(model)
#%% Show random heatmap
x_test2=x_test.reshape(2500,56,56,3)

# go through first 100 samples
#
for i in range(0,50):
    # 28,28,3 (rgb)
    im = x_test2[i]
    # step into the function
    h = heatmap.explain_prediction_heatmap(im, nmasks=(3,5,7,9,11))

#%%  check proportion class labels
y_test[y_test==0].size/len(y_test)
y_test[y_test==1].size/len(y_test)
y_test[y_test==2].size/len(y_test)
y_test[y_test==3].size/len(y_test)
