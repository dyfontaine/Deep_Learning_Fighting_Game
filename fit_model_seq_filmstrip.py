# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:44:14 2017

@author: adidier

fits a sequential NN using the film strip images
"""
import pickle
import os
import numpy as np

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.models import load_model
    
with open('Z:/Deep_learning/Deep_Learning_Project/labels', "rb") as fy:
    y_all = pickle.load(fy)


filename = 'Z:/Deep_Learning/Deep_Learning_Project/im_array' 
with open(filename, "r") as fp:
    im_array = pickle.load(fp)

n_images = len(im_array)
rows = 56
columns = 224    
film_strip_im = np.ndarray((n_images-4, rows, columns, 3), dtype=np.uint8)   


for i in range(3,n_images-1):   
   film_strip_im[[i-3]][0] = np.concatenate((im_array[[i-3]][0],im_array[[i-2]][0], \
                          im_array[[i-1]][0], im_array[[i]][0]),axis=1)
   
y = y_all[4:len(y_all)]   

n_images = len(film_strip_im)
rows = 56
columns = 224
x_all_flat = film_strip_im.reshape(n_images, rows*columns*3)

x_train, x_test = x_all_flat[:10000,], x_all_flat[10000:,]
y_train, y_test = y[:10000], y[10000:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size = 128
num_classes = 4
epochs = 20

# convert class vectors to binary class matrices
y_train2 = keras.utils.to_categorical(y_train, num_classes)
y_test2 = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(56*56*3,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.00001),
              metrics=['accuracy'])
history = model.fit(x_train, y_train2,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test2))
#%%
score = model.evaluate(x_test, y_test2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
#print confusion matrix

from sklearn.metrics import confusion_matrix

pClasses = model.predict_classes(x_test, batch_size=32, verbose=0)
probs = model.predict_proba(x_test, batch_size=32, verbose=0)

# top 2 accuracy?
#function to get top-k accuracy
def top_k_acc(y_actual,y_prob,k):
    cor = 0
    for row in range(0,len(y_actual)):
        predicted = y_prob[row,].argsort()[-k:][::-1]
        if y_actual[row] in predicted:
            cor += 1
    return(cor/len(y_actual))

print(top_k_acc(y_test,probs,2))