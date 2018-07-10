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
import math

def gaussian(x, c):
    xd = float(x)
    cd = float(c)
    s1 = 2.0
    s2 = 2.0
    c1 = cd-3.0
    c2 = cd+3.0
    return (-(xd-c2)*(xd-c2)/(2.0*s2*s2))# - math.exp(-(x-c1)*(x-c1)/(2*s1*s1)))

if True:
    history = {}
    epoch = []
    f = open("./data/output/AttentionConv-ORA-CNN5_64-RNN_256/metrics.pk",'rb')
    history = pickle.loads(f.read())
    init_epoch = len(history['loss']) - len(history['val_loss'])
    p = history['val_categorical_accuracy']
    max = 0
    amax = 0
    for i in range (len(p)):
        if p[i] > max:
            max = p[i]
            amax = i
    print(amax, max)
    f.close()
    df=pd.DataFrame({'abs': epoch, 'train_loss': history['loss'][len(history['val_loss']):]})#, 'val_loss': history['val_loss']})#, 'train_acc': history['categorical_accuracy'], 'val_acc': history['val_categorical_accuracy']})
    # multiple line plot
    #plt.subplot(2,1,1)
    #plt.plot( 'abs', 'val_loss', data=df, color='red', linewidth=2)
    plt.plot( 'abs', 'train_loss', data=df, color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function Value')
    plt.legend(loc=0)
    #plt.subplot(2,1,2)
    #plt.plot( 'abs', 'val_acc', data=df, color='red', linewidth=4)
    #plt.plot( 'abs', 'train_acc', data=df, color='blue', linewidth=4)
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0,1.0])
    #plt.legend(loc=0)
    plt.show("Figure 3")
else :
    for i in range(-10,10):
        print(gaussian(i,0))
    #cm.Model_Attention((28,28),(1,11), None)
