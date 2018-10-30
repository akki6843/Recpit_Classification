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


label = {
    0 : 'recepit',
    1 : 'non-recepit'
}

img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    x = Dense(1024, activation='relu')(model)
    predictions = Dense(2,activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model



def inference(imgname, label):
    model = create_model()
    model.load_weights('./model/recpit.h5')
    model.summary()
    img = cv2.imread(imgname)
    img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
    img = img.reshape(-1,224,224,3)
    prediction = model.predict(img)
    idx = np.argmax(prediction)
    return label[idx]

result = inference('./data/non-recepit/2.jpg', label)
print(result)

