# -*- coding: utf-8 -*-
''' '''
import os
import itertools
import codecs
import re
import datetime
import edistance
from edistance import edit_distance, edit_distance_backpointer
import numpy as np
import pylab
from pylab import *
import matplotlib
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.models import load_model
import keras.callbacks
import string
import glob
import cv2
from tqdm import tqdm
import network_helper as nt 
from network_helper import TextImageGenerator
import custom_model

def complete_states(stats, op, str1, str2):
    if (stats == []):
        stats = [[]]*11
        for i in range(11):
            stats[i] = [0]*11
    for i in range(len(op)):
        t = op[i]
        o = t[0]
        x = int(str1[t[1]])
        y = int(str2[t[3]])
        if (o=='equal'):
            stats[x][x] = stats[x][x] + 1
        if (o=='replace'):
            stats[y][x] = stats[y][x] + 1
        if (o=='insert'):
            stats[y][10] = stats[y][10] + 1
        if (o=='delete'):
            stats[10][x] = stats[10][x] + 1
    return stats

def test(run_name, img_w, img_h, start_epoch, use_ctc, use_att, minibatch_size, max_str_len, max_samples, batch_memory_usage, type_model, **kwargs):
    """
    Train a model

    #Argument
        - run_name: String, the name of the folder used as save directory for metrics, model and weights.
        - start_epoch: Int, the number corresponding to the weight_file that will be loaded
        - img_w : Int, Width of the input images  (All images need to be normalized)
        - img_h : Int, Height of the input images (All images need to be normalized)

    """

    if K.image_data_format() == 'channels_first':
        print('NOT IMPLEMENTED !!!')

    input_shape = (img_w, img_h)
    use_ctc = True
    print('Build text image generator')
    img_gen = TextImageGenerator(train_folder=datafolder_name,
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=4,
                                 val_split=0,
                                 alphabet=alphabet,
                                 absolute_max_string_len=max_str_len,
                                 max_samples=max_samples,
                                 acceptable_loss = 0,
                                 memory_usage_limit=batch_memory_usage,
                                 use_ctc=use_ctc,
                                 use_patches=True)
    minibatch_size = img_gen.minibatch_size
    nb_samples = img_gen.train_size
    input_shape = img_gen.input_shape
    print('Batch size : ',minibatch_size)
    act = 'relu'

    dir_path = os.path.join(OUTPUT_DIR,run_name)
    # model = load_model("./data/output/"+run_name+"/model.h5",custom_objects=custom_model.get_custom(type_model))
    # model.summary()
    model, test_f, _ = custom_model.get_model(type_model,input_shape,(max_str_len,len(alphabet)), img_gen, **kwargs)
    weight_file = os.path.join(dir_path,'weights%02d.h5' % (start_epoch-1))
    model.load_weights(weight_file, by_name=True)

    #print 'Compiling model'
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    input_data = model.get_layer('the_input').output
    y_pred1 = model.get_layer('the_output').output
    if test_f is None:
        test_func = [K.function([input_data], [y_pred1])]
    else:
        test_func = [test_f]
    if (use_ctc and use_att):
        y_pred2 = model.get_layer('the_output_ctc').output
        test_func = test_func + [K.function([input_data], [y_pred2])]

    #print(model.summary())
    step = nb_samples//minibatch_size
    print("Début prédiction")
    predict = model.predict_generator(generator=img_gen.next_train(),
                            steps=step,
                            max_queue_size=10,
                            workers=1)
    print("Début évaluation")
    N = 1
    accuracy_w = [0]*N
    accuracy_c = [0]*N
    nb_res = 0
    nb_mot = 0
    false_cpt = 0
    stats = []
    for i in range(step):
        if (use_ctc and not use_att):
            word_batch = next(img_gen.next_train())[0]
            num_proc = word_batch['the_input'].shape[0]
            if test_f is None:
            #decoded_res = nt.decode_batch(test_func[0],word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=True, n=N)[0]
            #print(decoded_res)
            #decoded_res2, _ = nt.decode_batch(test_func[1],word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=True, n=N)
            
            else :
                res = test_func[0]([word_batch['the_input'][0:num_proc], word_batch['input_length_decode'][0:num_proc]])
                proba = res[0]
                decoded_res = res[1:][0]
            for j in range(num_proc):
                dec_str = nt.labels_to_text(decoded_res[j], alphabet, 11)
                source_str = word_batch['source_str'][j]
                edit_dist, _ = edit_distance(dec_str, source_str) 
               # edit_dist2, _ = edit_distance(decoded_res2[j][0], source_str)
                if edit_dist == 0:# or edit_dist2 == 0:
                    acc = 1
                else :
                    acc = 0
                    #print(dec_str,source_str,edit_dist)
                accuracy_w[0] = accuracy_w[0] + acc
                accuracy_c[0] = accuracy_c[0] + min(edit_dist, len(source_str))
                nb_res = nb_res+1
                nb_mot = nb_mot + len(source_str)
        elif (use_ctc and use_att) :
            wb = next(img_gen.next_train())
            word_batch = wb[0]
            out_batch = wb[1]
            num_proc = word_batch['the_input'].shape[0]
            decoded_res1, scores1 = nt.decode_batch(test_func[0],word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=False, n=N)
            decoded_res2, scores2 = nt.decode_batch(test_func[1],word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=True, n=N)
            out = test_func[0]([word_batch['the_input'][0:num_proc]])[0]
            out2 = test_func[1]([word_batch['the_input'][0:num_proc]])[0]
            print(np.shape(out),np.shape(out2))
            for j in range(num_proc):
                smin = [100000000]*N
                amin = [-1]*N
                omin = [[]]*N
                source_str = nt.translate_array(out_batch['the_output'][j],alphabet, True)
                img = word_batch['the_input'][j].T
                for k in range(N):
                    print(source_str,decoded_res1[j][k],decoded_res2[j][k])
                    edit_dist1,_, ops1 = edit_distance_backpointer(decoded_res1[j][k], source_str)
                    edit_dist2,_, ops2 = edit_distance_backpointer(decoded_res2[j][k], source_str)
                    if edit_dist1 > edit_dist2:
                        edit_dist = edit_dist2
                        ops = ops2
                    else:
                        edit_dist = edit_dist1
                        ops = ops1
                    for l in range(k,N):
                        if (edit_dist<smin[l]):
                            smin[l] = edit_dist
                            amin[l] = k
                            omin[l] = ops
                #stats = complete_states(stats, omin[N-1],decoded_res1[j][amin[l]], source_str)
                for k in range(N):
                    edit_dist = smin[k]
                    if edit_dist > 0 :
                        acc = 0
                    else :
                        acc = 1
                    subplot(3,1,1)
                    imshow(img)
                    subplot(3,1,2)
                    att = out[j]
                    imshow(att.T, cmap=cm.hot)
                    subplot(3,1,3)
                    ctc = out2[j]
                    imshow(ctc.T, cmap=cm.hot)
                    show()
                    accuracy_w[k] = accuracy_w[k] + acc
                    accuracy_c[k] = accuracy_c[k] + min(edit_dist, len(source_str))
                nb_res = nb_res+1
                nb_mot = nb_mot + len(source_str)
        else:
            wb = next(img_gen.next_train())
            word_batch = wb[0]
            out_batch = wb[1]
            num_proc = word_batch['the_input'].shape[0]
            decoded_res, scores = nt.decode_batch(test_func[0],word_batch['the_input'][0:num_proc],alphabet, False, ctc_decode=use_ctc, n=N)
            for j in range(num_proc):
                smin = [100000000]*N
                amin = [-1]*N
                omin = [[]]*N
                source_str = nt.translate_array(out_batch['the_output'][j],alphabet, True)
                for k in range(N):
                    edit_dist,_, ops = edit_distance_backpointer(decoded_res[j][k], source_str)
                    for l in range(k,N):
                        if (edit_dist<smin[l]):
                            smin[l] = edit_dist
                            amin[l] = k
                            omin[l] = ops
                stats = complete_states(stats, omin[N-1],decoded_res[j][amin[l]], source_str)
                for k in range(N):
                    edit_dist = smin[k]
                    if edit_dist > 0 :
                        acc = 0
                    else :
                        acc = 1
                    accuracy_w[k] = accuracy_w[k] + acc
                    accuracy_c[k] = accuracy_c[k] + min(edit_dist, len(source_str))
                nb_res = nb_res+1
                nb_mot = nb_mot + len(source_str)
    for k in range(N):
        accuracy_w[k] = (accuracy_w[k]+0.0) / (nb_res+0.0)
        accuracy_c[k] = (nb_mot - accuracy_c[k]+0.0) / (nb_mot+0.0)
        print('Top ',k+1)
        print('Precision mot : ',accuracy_w[k])
        print('Precision caractere : ',accuracy_c[k])
    #print(stats)


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
    #datafolder_name = "../Dataset/MNIST/MNIST_Training_Multi"
    #datafolder_name = "../Dataset/ORAND-CAR/Binarized_CAR-A/a_train_images/"
    #datafolder_name = "../Dataset/CVL/CVLS_Training"
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
        print(kwargs)
        use_att = True
    else:
        kwargs = {}
        use_att = False
    if "CTC" in type_model:
        use_ctc=True
    else:
        use_ctc = False
    kwargs["loss"]=keras.losses.get('mse')
    kwargs["opt"]=keras.optimizers.get('sgd')
    test(run_name=run_name,start_epoch=start, type_model=type_model, use_ctc=use_ctc, use_att=use_att,
            img_w=image_width, img_h=image_height, minibatch_size=minibatch_size,
            max_str_len=max_str_len,max_samples=max_samples,batch_memory_usage=batch_memory_usage, **kwargs)

# ###            
