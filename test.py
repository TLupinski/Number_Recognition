import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import string
import glob
import cv2
import pydot
from tqdm import tqdm
import time
import pickle

if True:
    history = {}
    epoch = []
    f = open("./data/output/CConvEDAttC2:ORAND-A/metrics.pk",'r')
    history = pickle.loads(f.read())
    init_epoch = len(history['loss'])
    for i in range (init_epoch):
        epoch.append(i)
    f.close()
    df=pd.DataFrame({'abs': epoch, 'train_loss': history['loss'], 'val_loss': history['val_loss'], 'train_acc': history['categorical_accuracy'], 'val_acc': history['val_categorical_accuracy']})
    # multiple line plot
    plt.subplot(2,1,1)
    plt.plot( 'abs', 'val_loss', data=df, color='red', linewidth=2)
    plt.plot( 'abs', 'train_loss', data=df, color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function Value')
    plt.ylim([0,0.1])
    plt.legend(loc=0)
    plt.subplot(2,1,2)
    plt.plot( 'abs', 'val_acc', data=df, color='red', linewidth=4)
    plt.plot( 'abs', 'train_acc', data=df, color='blue', linewidth=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,1.0])
    plt.legend(loc=0)
    plt.show("Figure 3")
else :
    a = range(1,33)
    b = [a]
    c = [b,b,b]
    v = [c,c,c]
    rsres = np.zeros(shape=(32,3,3))
    for w in range(32):
        for x in range(3):
            for y in range(3):
                rsres[w][x][y] = v[x][y][0][w]
    print(v)
    print(rsres)
    #cm.Model_Attention((28,28),(1,11), None)
