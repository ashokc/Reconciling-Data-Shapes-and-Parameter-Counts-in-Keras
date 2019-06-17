import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(filters=55, kernel_size=(3, 2), padding="valid", strides=(2, 1), input_shape=(100, 95, 3),name='Conv_Layer_1'))
model.add(Conv2D(filters=35, kernel_size=(5, 2), padding="same", strides=(3, 2),name='Conv_Layer_2'))

from keras.utils import plot_model
model.summary()
plot_model(model, show_shapes=True, to_file='conv2d.png')
