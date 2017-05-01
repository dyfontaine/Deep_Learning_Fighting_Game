'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function

#%% ------ GPU memory fix -------
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())
#%% ------ GPU memory fix -------
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 50

# input image dimensions rows(height) columns (width)
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
# the data is originally in greyscale
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
# shape (no samples, no rows, no columns,greyscale dimension)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,               # set input mean to 0 over the dataset
    samplewise_center=False,                # set each sample mean to 0
    featurewise_std_normalization=False,    # divide inputs by std of the dataset
    samplewise_std_normalization=False,     # divide each input by its std
    zca_whitening=False,                    # apply ZCA whitening
    rotation_range=5,                       # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,                   # randomly flip images
    vertical_flip=False)                     # randomly flip images

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, shuffle=True), 
          epochs=epochs,
          verbose=1,
          steps_per_epoch=1,          
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Heatmaps
import numpy as np, matplotlib.pyplot as plt, skimage.exposure, skimage.color, skimage.filters, tqdm
def hide_axes(ax): ax.set_xticks([]), ax.set_yticks([])
class_index = {str(i): str(i) for i in range(10)}
# 10 is for digits 0 through 9

def predict(x):
    #1,28,28,1 made greyscale
    x = np.expand_dims(skimage.color.rgb2gray(x), -1)
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
            for c in range(5):
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
        for i, p in enumerate(predidx[:5]):
            print(p, preds[p], class_index[str(p)])
            topclasses[i] = (p, class_index[str(p)])
        # top class for index is key (index) value (class prediction)    
        print('Top classes: ', topclasses)
        
        # (5,28,28)= (5,) + (28,28)
        heatmaps = np.zeros((5,) + im.shape[1:3])
        self.predict_masks(im, masks, heatmaps, topclasses)
        for h in heatmaps: h = h / masknorm
        fig, axes = plt.subplots(2, 6, figsize=(10, 5))
        axes[0,0].imshow(img), axes[1,0].imshow(img)        
        axes[0,0].set_title(title)
        hide_axes(axes[0,0]), hide_axes(axes[1,0])       
        predictions = np.sum(heatmaps, axis=(1,2,))
        predictions /= predictions.max()
        for n, i in enumerate(np.argsort(predictions)[::-1][:5]):
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
i = np.random.randint(len(x_test))

# go through first 100 samples
#
for i in range(0,10):
    # 28,28,3 (rgb)
    im = skimage.color.gray2rgb(x_test[i].squeeze())
    # step into the function
    h = heatmap.explain_prediction_heatmap(im, nmasks=(3,5,7,9,11))

