import os
import sys
import itertools
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from scipy import ndimage
import pylab
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import string
import glob
import cv2
import pydot
from tqdm import tqdm
import time
import pickle
import math
import sklearn
from skimage.util import view_as_windows

def gaussian(x, c):
    xd = float(x)
    cd = float(c)
    s1 = 2.0
    s2 = 2.0
    c1 = cd-3.0
    c2 = cd+3.0
    return (-(xd-c2)*(xd-c2)/(2.0*s2*s2))# - math.exp(-(x-c1)*(x-c1)/(2*s1*s1)))

if False:
    history = {}
    epoch = []
    f = open("./data/output/KK2/metrics.pk",'rb')
    history = pickle.loads(f.read())
 #   init_epoch = len(history['loss']) - len(history['val_loss'])
    for i in range(len(history['loss'])):
	epoch = epoch+[i]
    #p = history['val_categorical_accuracy']
    f.close()
    print(len(epoch))
    print(len(history['loss']))
#    print(len(history['val_loss']))
    df=pd.DataFrame({'abs': epoch, 'train_loss': history['loss']}) #, 'val_loss': history['val_loss']}) #, 'train_acc': history['categorical_accuracy'], 'val_acc': history['val_categorical_accuracy']})
    # multiple line plot
    #plt.subplot(2,1,1)
#    plt.plot( 'abs', 'val_loss', data=df, color='red', linewidth=2)
    plt.plot( 'abs', 'train_loss', data=df, color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function Value')
    plt.ylim([0,60])
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
    im_path = "./DATASET/ORAND/Normalized_CAR-A/a_train_images/a_car_000154.png"#./DATASET/MNISTMulti1/MNISTM_Training/MNISTMult_13465.png"
    img = cv2.imread(im_path, 0)
    img_patched = view_as_windows(img, (32,20), step = 10)
    img_patched = img_patched[0]
    nbp = np.shape(img_patched)[0]
    plt.subplot2grid((2, nbp), (0, 0), colspan=nbp)
    plt.imshow(img, cmap=cm.gray)
    for i in range(nbp):
        plt.subplot2grid((2,nbp), (1, i))
        plt.imshow(img_patched[i], cmap=cm.gray)
    plt.show()



