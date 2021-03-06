# -*- coding: utf-8 -*-
''' '''
import os
import threading
import pickle
import numpy as np
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.callbacks import Callback
import itertools
import string
import glob
import cv2
from tqdm import tqdm
from keras.utils import plot_model
from skimage.util import view_as_windows
from math import ceil

t = 0

def noisy(noise_typ,image):
    if noise_typ == "gauss":
      shape = image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,shape)
      if len(image.shape)>2:
        gauss = gauss.reshape(shape[0],shape[1],shape[2])
      else:
        gauss = gauss.reshape(shape[0],shape[1])
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      s_vs_p = 0.5
      amount = 0.03
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def patching_images(img, patch_size, patch_step):
    pat = view_as_windows(img, patch_size, step = patch_step)[:,0]
    return pat

def translate_array(array,alphabet, del_spaces=False):
    """
    Translate function from classification array to string

    #Argument
        array: Classification array with L entries where each entry consist in an array with size=alphabet with one 1 and all other 0.
        alphabet: dictionnary used to convert from integer to char

    #Output
        res: Corresponding string
    """
    res = ''
    for i in range(len(array)):
        n = np.argmax(array[i])
        if (array[i][n] != 0):
            if not(del_spaces and alphabet[n]==' '):
                res = res + (alphabet[n])
    return res

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

def text_to_labels(text, alphabet, size=-1):
    """Translation of characters to unique integer values."""
    if (size==-1):
        ret = []
        for char in text:
            ret.append(alphabet.find(char))
        return ret
    else:
        ret = np.zeros((size,len(alphabet)))
        i = 0
        for char in text:
            n = (alphabet.find(char))
            ret[i][n] = 1
            i = i + 1
        for i in range(len(text),size):
            ret[i][len(alphabet)-1] = 1
        return ret

def labels_to_text(labels, alphabet, size):
    """Reverse translation of numerical classes back to characters."""
    ret = []
    for c in labels:
        if c == size or c ==-1:  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

class TextImageGenerator(keras.callbacks.Callback):
    """
    Uses generator functions to supply train/test with data.
    Images are loaded on the fly to keep RAM usage low

    #Argument
        train_folder:   folder containing all images
        img_w :     max image width
        img_h :     max image height
        val_split : % of the dataset used for validation
        alphabet :  all possible character that need to be recognized
        minibatch_size: define batch_size, if -1, optimal batch_size computed
        maxbatch_size:  define maximum batch size if minibatch_size = -1
        acceptable_loss:    Max number of samples that can be discarded from training set 
            for maximizing batch_size
        memory_usage_limit: Max RAM usage allowed for each batch
        absolute_max_string:Max length of output strings
        channels:   Number of channels in the images (1:Grayscale|3:RGB)
    """

    def __init__(self, train_folder, img_w, img_h, downsample_factor, val_split,
                 alphabet, use_ctc=False, minibatch_size=-1, maxbatch_size=-1, 
                 absolute_max_string_len=16, max_samples=1000, noise=None,
                 memory_usage_limit=2000000, acceptable_loss = 0, channels=1,
                 use_patches=False, patch_size = (12,32), patch_step=8):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        if use_patches:
            self.decode_length = patch_size[0]
        else:
            self.decode_length = img_w/downsample_factor
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
        self.acceptable_loss = acceptable_loss
        self.use_ctc = use_ctc
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.lock = threading.Lock()
        self.noise = noise
        if maxbatch_size ==-1:
            self.maxbatch_size = int(memory_usage_limit / (img_w*img_h))
        if use_patches:
            step = img_w/patch_step - int(ceil(float((patch_size[0]-patch_step))/float(patch_step)))
            self.input_shape = (step, patch_size[0], patch_size[1])
        else : 
            self.input_shape = (img_w, img_h)
        if channels == 1:
            self.readmode = 0
        else :
            self.readmode = 1
            self.input_shape = sef.input_shape + (channels,)
        print('Build image list...')
        self.build_image_list()
        print('Split train/val set')
        self.split_train_set()

    #Load all images names and ground truth
    def build_image_list(self):
        search_path = os.path.join(self.train_folder,"*.png")
        lines = glob.glob(search_path)

        with tqdm(total=len(lines)) as pbar:
            for i, line in enumerate(lines):
                if len(self.train_data) >= self.max_samples:
                    break
                text_file = "{}txt".format(line[:-3])
                text = self.get_text(text_file)
                if len(text) > 0 and len(text) <= self.absolute_max_string_len:
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

    #Split train and validation set using val_split
    def split_train_set(self):
        self.val_size = int(self.train_size*self.val_split)
        self.train_size = self.train_size - self.val_size

        self.val_data = self.train_data[self.train_size:]
        self.train_data = self.train_data[:self.train_size]
        print("#{} training samples before |#{} validation samples".format(len(self.train_data),len(self.val_data)))

        #If no batch_size is specified, calculate batch_size = pgcd(val_size,val_data)<max_batch_size
        if (self.minibatch_size == -1):
            max = 0
            submax = 0
            for i in range(1,self.maxbatch_size + 1):
                if (self.val_size%i<1)&(self.train_size%i<1):
                    max = i
                if (self.val_size%i<=self.acceptable_loss)&(self.train_size%i<=self.acceptable_loss):
                    submax = i
            if (submax > max):
                max = submax
            self.minibatch_size = max
        # Make each set size // minibatch size == 0
        self.train_data, self.train_size = self.make_it_mod_zero(self.train_data)
        self.val_data, self.val_size =self.make_it_mod_zero(self.val_data)
        print("#{} training samples | #{} validation samples".format(self.train_size, self.val_size))


    def get_output_size(self):
        return len(self.alphabet) + 1

    #Load one image from training set
    def get_train_image(self, index):
        im_path = self.train_data[index]['image']
        if os.path.exists(im_path):
            img = cv2.imread(im_path, self.readmode)
            if (self.noise!=None):
                img = noisy(self.noise,img)
            img = np.divide(np.transpose(img ,(1,0)),127.5)-1.0
            if (self.use_patches):
                img = patching_images(np.ascontiguousarray(img), self.patch_size, self.patch_step)
            return img
        else:
            print("File not found {}".format(self.train_data[index]))
        return ""
    #Return ground truth from training set
    def get_train_text(self, index):
        return self.train_data[index]['text']

    #Load one image from validation set
    def get_val_image(self, index):
        im_path = self.val_data[index]['image']
        if os.path.exists(im_path):
            img = cv2.imread(im_path, self.readmode)
            img = np.divide(np.transpose(img ,(1,0)),127.5)-1.0
            if (self.use_patches):
                img = patching_images(np.ascontiguousarray(img), self.patch_size, self.patch_step)
            return img
        else:
            print("File not found {}".format(self.train_data[index]))
        return ""

    #Return ground truth from validation set
    def get_val_text(self, index):
        return self.val_data[index]['text']

    #Read a text from a file and return the extracted line
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


    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones((size,) + self.input_shape)
        else:
            X_data = np.ones((size,) + self.input_shape)

        if self.use_ctc:
            input_length = np.zeros([size, 1])
            label_length = np.zeros([size, 1])
            source_str = []
            labels_ctc = np.ones([size, self.absolute_max_string_len])
            labels = np.zeros([size, self.absolute_max_string_len, len(self.alphabet)])
        else:
            labels = np.zeros([size, self.absolute_max_string_len, len(self.alphabet)])
        
        if K.image_data_format() == 'channels_first':
            print('Not IMPLEMENTED !')

        for i in range(size):
            text = ""
            if train:
                X_data[i] = self.get_train_image(index+i)
                text = self.get_train_text(index+i)
            else:
                X_data[i] = self.get_val_image(index+i)
                text = self.get_val_text(index+i)
            if self.use_ctc:
                input_length[i] = self.decode_length
                label_length[i] = len(text)
                source_str.append(text)
                labels_ctc[i, 0:len(text)] = text_to_labels(text, self.alphabet, -1)
                labels[i] = text_to_labels(text, self.alphabet, self.absolute_max_string_len)
            else:
                labels[i] = text_to_labels(text, self.alphabet, self.absolute_max_string_len)
        X_data = np.array(X_data)
        decode_length = np.reshape(input_length,[size])
        if self.use_ctc:
            inputs = {'the_input': X_data,
                      'the_labels': labels_ctc,
                      'input_length': input_length,
                      'input_length_decode': decode_length,
                      'label_length': label_length,
                      'source_str': source_str}  # used for visualization only
            outputs = {'the_output': labels ,
                      'the_output_ctc': labels,#np.zeros((size,128,11)) ,
                      'ctc': np.zeros([size,1])}  # dummy data for dummy loss function
        else:
            labels = np.array(labels)
            inputs = {'the_input': X_data}
            outputs = {'the_output': labels }
        return (inputs, outputs)

    def get_val_steps(self):
        return self.val_size // self.minibatch_size

    def get_train_steps(self):
        return self.train_size // self.minibatch_size

    def next_train(self):
        with self.lock:
            while 1:
                ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
                self.cur_train_index += self.minibatch_size
                if self.cur_train_index >= self.train_size:
                    self.cur_train_index = self.cur_train_index % self.minibatch_size
                yield ret

    def next_val(self):
        with self.lock:
            while 1:
                ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
                self.cur_val_index += self.minibatch_size
                if self.cur_val_index >= self.val_size:
                    self.cur_val_index = self.cur_val_index % self.minibatch_size
                yield ret

    def on_train_begin(self, logs={}):
        print('Begin training...')

    def on_epoch_begin(self, epoch, logs={}):
        #Shuffle all samples in the training set for batch variations
        np.random.shuffle(self.train_data)
        print('Epoch begin..')


def ctc_lambda_func(args):
    """
    The actual loss calc occurs here despite it not being
    an internal Keras loss function.
    """
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_lambda_decode_func(args):
    """
    CTC output lambda function to use as a Lambda layer in a model
    """
    y_pred, input_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_decode(y_pred, input_length, greedy=False,beam_width=20)

def decode_batch(test_func, word_batch,alphabet, display=False, ctc_decode=False, n=1):
    """
    For a real OCR application, this should be beam 
    search with a dictionary and language model. 
    For this example, best path is sufficient.
    """
    out = test_func([word_batch])[0]
    # if display:
    #     for l in range(1):
    #         k = 5
    #         for i in range(k):
    #             b = word_batch[l*k + i]
    #             b = swapaxes(b,0,1)
    #             subplot(k,2,2*i+1)
    #             imshow(b)
    #             subplot(k,2,2*i+2)
    #             imshow(out[l*k + i].T,cmap=cm.hot)
    #         show()
    ret = []
    for i in range(out.shape[0]):
        ret.append([])
    for j in range(out.shape[0]):
        dx = 1
        if (ctc_decode):
            dx = 0
        if n == 1:
            out_best = list(np.argmax(out[j], 1))
            if ctc_decode:
                out_best = list(np.argmax(out[j], 1))
                out_best = [k for k, g in itertools.groupby(out_best)]
            scores = [1]
            outstr = labels_to_text(out_best,alphabet, len(alphabet)-dx)
            ret[j].append(outstr)
        else:
            out_best, scores = decode_n_best(out[j], n, len(alphabet))
            if ctc_decode:
                new_out_best = []
                for i in range(n):
                    new_out_best = new_out_best + [[k for k, g in itertools.groupby(out_best[i])]]
                out_best = new_out_best
            for i in range(n):
                ret[j].append(labels_to_text(out_best[i],alphabet, len(alphabet)-dx))
    # if ctc_decode:
    #     input_length = [32]*out.shape[0]
    #     out_best, scores = K.ctc_decode(out, input_length, greedy = False, top_paths = n)
    #     print(out_best)
    #     print(scores)
    #     for j in range(out.shape[0]):
    #         print(out_best[0][j], ret[j])

    return ret, scores

def decode_n_best(res, n, a):
    score = np.zeros((n))
    best = np.flip(np.argsort(res[0]),0)
    label = np.zeros((n,len(res)),dtype=np.int32)
    labels = np.zeros((n,len(res)),dtype=np.int32)
    for i in range(n):
        label[i][0] = int(best[i])
        score[i] = res[0][label[i][0]]
    for i in range(1,len(res)):
        scores = np.zeros((n*a))
        for j in range(n):
            for k in range(a):
                scores[j*a + k] = score[j] * res[i][k]
        best = np.flip(np.argsort(scores),0)
        for j in range(n*a-1):
            assert scores[best[j]]>=scores[best[j+1]]
        for j in range(n):
            m = best[j]
            af = m%a
            bf = (m-af)//a
            labels[j] = label[bf] 
            labels[j][i] = af
        for j in range(n):
            label[j] = labels[j]
            score[j] = scores[best[j]]
    return label, score
