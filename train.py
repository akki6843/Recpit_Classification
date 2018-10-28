import numpy as np
import cv2
import keras
from keras.datasets import fashion_mnist 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt

img_rows, img_cols = 200,1000

X_train = 0
Y_train = 0



input_shape = (img_rows, img_cols, 3)
def create_model():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (1, 1), activation='relu'))

    model.add(Dense(10, activation='softmax'))
    return model

model = create_model()
model.summary()

def train():
    model = create_model()
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=1,shuffle=False, validation_split=0.3)
    model.save_weights('./model/recpit.h5')

# train()
