# -*- coding: utf-8 -*-
''' '''
import os
import itertools
import codecs
import re
import datetime
import pickle
import cairocffi as cairo
import editdistance
import pickle
import numpy as np
from scipy import ndimage
import pylab
from pylab import *
import matplotlib
from keras import backend as K
from keras.backend import tf
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from keras.callbacks import History
import string
import glob
import cv2
import pydot
from tqdm import tqdm
from keras.utils import plot_model

t = 0

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret

def text_to_labels(text, alphabet):
    """Translation of characters to unique integer values."""
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

def labels_to_text(labels, alphabet):
    """Reverse translation of numerical classes back to characters."""
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

class TextImageGenerator(keras.callbacks.Callback):
    """
    Uses generator functions to supply train/test with data.
    Images are loaded on the fly.
    """

    def __init__(self, train_folder, img_w, img_h, 
                 downsample_factor, val_split, alphabet, minibatch_size=-1, maxbatch_size=-1, absolute_max_string_len=16, max_samples=1000):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.alphabet = alphabet
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.train_folder = train_folder
        self.train_data = []
        self.val_data = []
        self.train_size = 0
        self.val_size = 0
        self.max_samples = max_samples
        self.maxbatch_size = maxbatch_size
        if maxbatch_size ==-1:
            self.maxbatch_size = 1000000 / (img_w*img_h)

        print 'Build image list...'
        self.build_image_list()
        print 'Split train/val set'
        self.split_train_set()

    def build_image_list(self):
        search_path = os.path.join(self.train_folder,"*.png")
        lines = glob.glob(search_path)

        with tqdm(total=len(lines)) as pbar:
            for i, line in enumerate(lines):
                if len(self.train_data) >= self.max_samples:
		    break
                text_file = "{}txt".format(line[:-3])
                text = self.get_text(text_file)
                if len(text) > 0 and len(text) < self.absolute_max_string_len:
                        self.train_data.append({'image':line,'text':text})
                	pbar.update()
        pbar.close()
        self.train_size = len(self.train_data)

        self.cur_val_index = 0
        self.cur_train_index = 0

    def make_it_mod_zero(self, array):
        size = len(array)
        size_offset = size % self.minibatch_size
        new_array = array
        if size_offset > 0:
            new_array = array[:-size_offset]
        new_size = size - size_offset
        assert(new_size == len(new_array))
        assert((new_size % self.minibatch_size)==0)
        return new_array, new_size

    def split_train_set(self):
        self.val_size = int(self.train_size*self.val_split)
        self.train_size = self.train_size - self.val_size

        self.val_data = self.train_data[self.train_size:]
        self.train_data = self.train_data[:self.train_size]
	print("#{} training samples before |#{} validation samples".format(len(self.train_data),len(self.val_data)))

        #If no batch_size is specified, calculate batch_size = pgcd(val_size,val_data)<500
        if (self.minibatch_size == -1):
            max = 0
            for i in range(1,self.maxbatch_size + 1):
                if (self.val_size%i<1)&(self.train_size%i<1):
                    max = i
            self.minibatch_size = max
        # Make each set size // minibatch size == 0
        self.train_data, self.train_size = self.make_it_mod_zero(self.train_data)
        self.val_data, self.val_size =self.make_it_mod_zero(self.val_data)
        print("#{} training samples | #{} validation samples".format(self.train_size, self.val_size))
        
    def get_output_size(self):
        return len(self.alphabet) + 1

    def get_train_image(self, index):
        im_path = self.train_data[index]['image']
        if os.path.exists(im_path):
            return np.divide(np.transpose(np.mean(cv2.imread(im_path),axis=2),(1,0)),255.)
        else:
            print "File not found {}".format(self.train_data[index])
        return ""

    def get_train_text(self, index):
        return self.train_data[index]['text']

    def get_val_image(self, index):
            return np.divide(np.transpose(np.mean(cv2.imread(self.val_data[index]['image']),axis=2),(1,0)),255.)

    def get_text(self, file_path):
        fil = open(file_path,"r")
        text = ""
        for line in fil:
            line = line.lower()
            for c in line:
                if c in self.alphabet:
                    text += c
            return text
        return text

    def get_val_text(self, index):
        return self.val_data[index]['text']

    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 3, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 3])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        
        if K.image_data_format() == 'channels_first':
            print 'Not IMPLEMENTED !'

        for i in range(size):
            text = ""
            if train:
                X_data[i] = self.get_train_image(index+i)
                text = self.get_train_text(index+i)
            else:
                X_data[i] = self.get_val_image(index+i)
                text = self.get_val_text(index+i)
            labels[i, 0:len(text)] = text_to_labels(text, self.alphabet)
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = len(text)
            source_str.append(text)

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def get_val_steps(self):
        return self.val_size // self.minibatch_size

    def get_train_steps(self):
        return self.train_size // self.minibatch_size

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.train_size:
                self.cur_train_index = self.cur_train_index % self.minibatch_size
                # Skip the shuffle in the first place
                # (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                #     [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.val_size:
                self.cur_val_index = self.cur_val_index % self.minibatch_size
            yield ret

    def on_train_begin(self, logs={}):
        print('Begin training...')

    def on_epoch_begin(self, epoch, logs={}):
        print('Epoch begin..')

def ctc_lambda_func(args):
    """
    The actual loss calc occurs here despite it not being
    an internal Keras loss function.
    """
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_lambda_decode_func(args):
    """
    The actual loss calc occurs here despite it not being
    an internal Keras loss function.
    """
    y_pred, input_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    return K.ctc_decode(y_pred, input_length, greedy=False,beam_width=20)

def decode_batch(test_func, word_batch,alphabet, display=False):
    """
    For a real OCR application, this should be beam 
    search with a dictionary and language model. 
    For this example, best path is sufficient.
    """
    out = test_func([word_batch])[0]
    print('YOLO')
    print(out)
    if display:
        k = 3
        for i in range(k):
            b = word_batch[i]
            b = swapaxes(b,0,1)
            subplot(k,2,2*i+1)
            imshow(b)
            subplot(k,2,2*i+2)
            imshow(out[i].T,cmap=cm.hot)
        show()
    ret = []
    for j in range(out.shape[0]):
        print(j)
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        print(out_best)
        outstr = labels_to_text(out_best,alphabet)
        print(outstr)
        ret.append(outstr)
    return ret