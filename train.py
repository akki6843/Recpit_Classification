'''
This is the training script for making the recepit classifier
'''

#Importing libraries 

import glob
import os
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import random
import pdb


'''
# this part of the code is for restricting the gpu to a certain percentage
import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session 
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.3 
set_session(tf.Session(config=config))
'''

# Getting the path of dataset
paths = "./Recpit_Classification/data/*/*"

# Collecting all the path
data_path = glob.glob(paths)
print(len(data_path))

# Data_Spinner is the suncation that will generate the data ready for neural network
def data_spinner(data_path):
	random.shuffle(data_path)
	Y = []
	X = []
	for img_path in data_path:
		img = cv2.imread(img_path)
		img = cv2.resize(img, (224,224))
		img1 =  np.array(img,dtype='float')/255.0
		X.append(img1)
		name = img_path.split("/")
		if 'recepit' in name:
			Y.append([1,0])
		if 'non-recepit' in name:
			Y.append([0,1])
	Y = np.array(Y)
	#pdb.set_trace()
	X = np.array(X)
	return X,Y
		
			

X, Y = data_spinner(data_path)

print(X)


img_rows, img_cols = 224, 224

# Creating model for training

input_shape = (img_rows, img_cols, 3)
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    x = Dense(1024, activation='relu')(model)
    predictions = Dense(2,activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

# model = create_model()
# model.summary()

# Training opreation with validation on 30% of the data
def train():
    model = create_model()
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, batch_size=16, epochs=30, verbose=1,shuffle=True, validation_split=0.3)
    model.save_weights('./model/recpit.h5')

train()
