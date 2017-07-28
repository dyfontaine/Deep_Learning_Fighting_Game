from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.models import load_model

import cv2
import pandas as pd
import numpy as np
import glob

size = 56,56
rows, columns = 56, 56
n_images = 12500

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

#%%
#MAIN
batch_size = 128
num_classes = 4
epochs = 20

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

x_all_flat = x_all.reshape(n_images, rows*columns*3)
x_train, x_test = x_all_flat[:10000,], x_all_flat[10000:,]
y_train, y_test = y_all[:10000], y_all[10000:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#%%
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

#save the model
model.save('model1_fighting.h5')


#%%
#look at model performance
score = model.evaluate(x_test, y_test2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
#function to get top-k accuracy
def top_k_acc(y_actual,y_prob,k):
    cor = 0
    for row in range(0,len(y_actual)):
        predicted = y_prob[row,].argsort()[-k:][::-1]
        if y_actual[row] in predicted:
            cor += 1
    return(cor/len(y_actual))

print(top_k_acc(y_test,probs,2))

