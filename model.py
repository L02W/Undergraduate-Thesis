from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, mobilenet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from args import Args
from pathlib import Path
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import argparse
import os
#import cv2

class Models:

    def __init__(self):
        pass

    def vgg16(self):
        baseModel = Input(shape=(512,))

        headModel = Dense(512, activation='relu')(baseModel)
        headModel = Dropout(Args.dropout)(headModel)
        headModel = Dense(2, activation='softmax', name='softmax')(headModel)
        model = Model(inputs=baseModel, outputs=headModel, name='vgg16')

        model.compile(loss="categorical_crossentropy", optimizer='adam',
                      metrics=["accuracy"])

        return model


    def inception(self):
        baseModel = Input(shape=(2048,))
        #baseModel = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        headModel = Dense(512, activation='relu')(baseModel)
        headModel = Dropout(Args.dropout)(headModel)
        headModel = Dense(2, activation='softmax', name='softmax')(headModel)
        model = Model(inputs=baseModel, outputs=headModel, name='inception')

        model.compile(loss="categorical_crossentropy", optimizer='adam',
                      metrics=["accuracy"])

        return model

    def resnet(self):
        baseModel = Input(shape=(2048,))
        #baseModel = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        headModel = Dense(512, activation='relu')(baseModel)
        headModel = Dropout(Args.dropout)(headModel)
        headModel = Dense(2, activation='softmax', name='softmax')(headModel)
        model = Model(inputs=baseModel, outputs=headModel, name='resnet')

        model.compile(loss="categorical_crossentropy", optimizer='adam',
                      metrics=["accuracy"])

        return model

    def mobilenet(self):
        baseModel = Input(shape=(1024,))
        #baseModel = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        headModel = Dense(512, activation='relu')(baseModel)
        headModel = Dropout(Args.dropout)(headModel)
        headModel = Dense(2, activation='softmax', name='softmax')(headModel)
        model = Model(inputs=baseModel, outputs=headModel, name='mobilenet')

        model.compile(loss="categorical_crossentropy", optimizer='adam',
                      metrics=["accuracy"])

        return model



