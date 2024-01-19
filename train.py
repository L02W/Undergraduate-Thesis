from utils import *
from model import Models
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from collections import Counter
from model import Models
import os
from args import Args
from utils import read_pickle
from keras.utils import np_utils
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging
import re
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from sklearn import svm
import os
from tensorflow.keras.applications import VGG16, ResNet50, Xception, MobileNet
from tensorflow.keras.layers import Input

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


logging.basicConfig(level=logging.INFO, format=('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


def train_model(model):

    logger = logging.getLogger(model.name)
    logger.info('train {}...'.format(model.name))

    csv_logger = CSVLogger('results/{}/train.csv'.format(model.name), append=True)
    es = EarlyStopping(monitor='val_loss', patience=5)
    mcp = ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True,
                          save_weights_only=True, filepath='results/{}/best.h5'.format(model.name))

    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=Args.batch_size, epochs=Args.epochs, verbose=1,
              callbacks=[csv_logger, es, mcp])
    model.load_weights(r'results/{}/best.h5'.format(model.name))
    pre = model.predict(testX, batch_size=Args.batch_size, verbose=1)
    pre_ = pre.argmax(axis=-1)
    testY_ = testY.argmax(axis=-1)

    report = classification_report(testY_, pre_, digits=5)
    return report


if __name__ == '__main__':

    imgs, labs = read_pickle(r'imgs.pickle'), read_pickle(r'labs.pickle')
    trainX, testX, trainY, testY = train_test_split(imgs, labs, test_size=0.2, random_state=42)

    myModel = Models().vgg16()
    baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)),
                      pooling='avg')
    trainX = baseModel.predict(trainX, batch_size=16)
    testX = baseModel.predict(testX, batch_size=16)

    '''
    myModel = Models().inception()
    baseModel = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)),
                         pooling='avg')
    trainX = baseModel.predict(trainX, batch_size=16)
    testX = baseModel.predict(testX, batch_size=16)

    myModel = Models().resnet()
    baseModel = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)),
                         pooling='avg')
    trainX = baseModel.predict(trainX, batch_size=16)
    testX = baseModel.predict(testX, batch_size=16)

    myModel = Models().mobilenet()
    baseModel = MobileNet(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)),
                         pooling='avg')
    trainX = baseModel.predict(trainX, batch_size=16)
    testX = baseModel.predict(testX, batch_size=16)
    '''


    if not os.path.exists('results/' + myModel.name):
        os.makedirs('results/' + myModel.name)

    logger = logging.getLogger(myModel.name)

    report = train_model(myModel)
    print(report)

