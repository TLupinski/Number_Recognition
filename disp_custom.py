# -*- coding: utf-8 -*-
''' '''
import os
import itertools
import codecs
import re
import datetime
import numpy as np
import pylab
from pylab import *
import matplotlib
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import string
import glob
import cv2
from tqdm import tqdm
import network_helper as nt 
from network_helper import TextImageGenerator
import custom_model

def test(run_name, img_w, img_h, start_epoch, minibatch_size, max_str_len, max_samples, batch_memory_usage, type_model, **kwargs):
    weight_file = "data/output/"+run_name+"/weights"+str(start_epoch-1)+".h5"

    if K.image_data_format() == 'channels_first':
        print('NOT IMPLEMENTED !!!')

    use_ctc = True
    input_shape = (img_w, img_h)
    print('Build text image generator')
    img_gen = TextImageGenerator(train_folder=datafolder_name,
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=4,
                                 val_split=val_split,
                                 alphabet=alphabet,
                                 absolute_max_string_len=max_str_len,
                                 max_samples=max_samples,
                                 acceptable_loss = 0,
                                 memory_usage_limit=5000000,
                                 use_ctc=use_ctc)
    minibatch_size = img_gen.minibatch_size
    nb_samples = img_gen.train_size
    print('Batch size : ',minibatch_size)
    act = 'relu'

    dir_path = os.path.join(OUTPUT_DIR,run_name)
    weight_file = os.path.join(dir_path,'weights%02d.h5' % (start_epoch-1))
    model, test_func, _ = custom_model.get_model(type_model,input_shape,(max_str_len,len(alphabet)), img_gen, weight_file, **kwargs)
    #print 'Compiling model'
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    input_data = model.get_layer('the_input').output
    #e_pred = model.get_layer('encoder').output
    y_predatt = model.get_layer('the_output').output
    y_predctc = model.get_layer('the_output_ctc').output
    #enc_func = K.function([input_data], [e_pred])
    out_func = K.function([input_data], [y_predatt,y_predctc])
    attention = True
    #print(model.summary())
    step = nb_samples//minibatch_size
    print("Début prédiction")
    predict = model.predict_generator(generator=img_gen.next_train(),
                            steps=step)
    print("Début scorring")
    accuracy_w = 0
    accuracy_c = 0
    nb_res = 0
    nb_mot = 0
    hidden = 16 * 2
    n_img_w = img_w//4
    n_img_h = img_h//16
    nb_display = 2
    for i in range(step):
        wb = word_batch = next(img_gen.next_train())
        word_batch = wb[0]
        out_batch = wb[1]
        num_proc = word_batch['the_input'].shape[0]
        #enc = enc_func([word_batch['the_input'][0:num_proc]])[0]
        out = out_func([word_batch['the_input'][0:num_proc]])[0]
        out2 = out_func([word_batch['the_input'][0:num_proc]])[1]
        #decoded_res, scores = nt.decode_batch(out_func,word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=use_ctc, n=1)
        print(np.shape(out), np.shape(out2))
        if attention:
            for i in range(minibatch_size):
                img = word_batch['the_input'][i].T
                shape = np.shape(img)
                img = np.repeat(np.reshape(img,shape + (1,)),3,axis=-1)
                subplot(3,1,1)
                imshow(img)
                subplot(3,1,2)
                att = out[i]
                imshow(att.T, cmap=cm.hot)
                subplot(3,1,3)
                ctc = out2[i]
                imshow(ctc.T, cmap=cm.hot)
                # for j in range(max_str_len):
                #     att = out[i][j][0]
                #     ctc = out[i][j][1]
                #     superposed = np.copy(img)
                #     for a in range(img_w-4):
                #         for b in range(img_h):
                #             n = 0
                #             if (att[a//8]>0):
                #                 x = 0.5 + superposed[b][a][0]/2 + att[a//8]
                #                 if (x > 1.0):
                #                     n = x - 1.0
                #                     x = 1.0
                #                 superposed[b][a][0] = x
                #             superposed[b][a][1] = 0.5 + superposed[b][a][1]/2 -n
                #             superposed[b][a][2] = 0.5 + superposed[b][a][2]/2 -n
                #     subplot(max_str_len+1,1,2+j)
                #    imshow(att.T, cmap=cm.hot)
                show()
        else:
            #Afficher les différentes couches de convolutions
            if (len(np.shape(out)) > 3):
                decoded_res = np.zeros(shape=(minibatch_size,hidden,n_img_w,n_img_h))
                for j in range(minibatch_size):
                    for w in range(hidden):
                        for x in range(n_img_w):
                            for y in range(n_img_h):
                                decoded_res[j][w][x][y] = out[j][x][y][w]
                for j in range(3):#minibatch_size):
                    for k in range(hidden):
                        b = decoded_res[j][k]
                        subplot(24, 8,k+1+j*hidden)
                        imshow(b.T)
                show()
            #Afficher les séquences comme des images
            else:
                display_size = 5
                out = np.reshape(out[:display_size], (display_size,n_img_w,11))
                for j in range(display_size):
                    img = word_batch['the_input'][j]
                    img = np.repeat(np.reshape(img,np.shape(img)+(1,)),3,axis=-1)
                    subplot(nb_display,display_size,j+1)
                    imshow(word_batch['the_input'][j].T, cmap=cm.gray)
                if nb_display == 3:
                    for j in range(display_size):
                        subplot(nb_display,display_size,j+1+display_size)
                        imshow(enc[j].T, cmap=cm.gray)
                for j in range(display_size):
                    subplot(nb_display,display_size,j+1+(nb_display-1)*display_size)
                    imshow(out[j].T, cmap=cm.hot)
                show()
    
if __name__ == '__main__':
    if (len(sys.argv)<2):
        init_file = open('init_test')
    else:
        init_file = open(sys.argv[1])
    init_content = init_file.readlines()

    for i in range(len(init_content)):
        init_content[i] = init_content[i].split('|')[1]
        init_content[i] = init_content[i].replace('\n','')


    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    OUTPUT_DIR = './data/output/'
    weight_file = "data/output/weight00.h5"
    run_name = init_content[0]
    datafolder_name = init_content[1]
    image_width = int(init_content[2])
    image_height = int(init_content[3])
    #Maximum length of output needed
    max_str_len = int(init_content[4])
    #Maximum number of samples used from the data folder
    max_samples = int(init_content[5])
    #Value ]0.0,1.0[ used to split validation set from training set. 0 is for no validation
    val_split = float(init_content[6])
    #Batch_size : use -1 to let the generator decide the optimal batch_size
    minibatch_size = int(init_content[7])
    #Amount of RAM memory you can afford for each batch, used only if minibatch_size is set to -1
    batch_memory_usage = int(init_content[8])
    start = int(init_content[9])
    stop = int(init_content[10])
    modelpath = ''
    # character classes
    alphabet = init_content[11] #+ string.lowercase +  " " # + string.uppercase + string.punctuation
    type_model=init_content[12]
    if "Attention" in type_model and len(init_content) >= 16:
        c1 = [int(x) for x in init_content[13].split(',')]
        c2 = [int(x) for x in init_content[14].split(',')]
        enc = [int(x) for x in init_content[15].split(',')]
        dec = [int(x) for x in init_content[16].split(',')]
        kwargs = {'CNN' : [c1,c2],
                'Encoder' : enc,
                'Decoder' : dec}
    else:
        kwargs = {}
    kwargs["loss"]='mse'
    kwargs["opt"]='sgd'
    test(run_name=run_name,start_epoch=start, type_model=type_model,
            img_w=image_width, img_h=image_height, minibatch_size=minibatch_size,
            max_str_len=max_str_len,max_samples=max_samples,batch_memory_usage=batch_memory_usage, **kwargs)

