import numpy as np
import keras
from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=(3, 2), padding="valid", strides=(2, 1),input_shape=(100, 95, 3),name='Valid_Pool_Layer_1'))
model.add(MaxPooling2D(pool_size=(5, 2), padding="same", strides=(3, 2),name='Same_Pool_Layer_2'))

from keras.utils import plot_model
model.summary()
plot_model(model, show_shapes=True, to_file='pooling.png')
