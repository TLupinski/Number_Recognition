# -*- coding: utf-8 -*-
''' '''
import os
import sys
import itertools
import codecs
import re
import datetime
import pickle
import numpy as np
import pylab
from keras import backend as K
from keras import regularizers
from keras.backend import tf
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
import keras.optimizers
import keras.losses
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import keras.losses
import keras.optimizers
from keras.callbacks import History
import cv2
import edistance
import custom_model
from tqdm import tqdm
from keras.utils import plot_model
import network_helper as nt
from network_helper import TextImageGenerator

class VizCallback(keras.callbacks.Callback):
    '''
    Callback utilisé pour:
        - tester les résultats obtenus sur l'ensemble de validation
        - sauvegarder les poids/le modèle entier à la fin de chaque époque
    '''

    def __init__(self, run_name, test_func, text_img_gen, num_edit=256, num_display_words=10):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_edit = num_edit
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            wb = next(self.text_img_gen)
            word_batch = wb[0]
            out_batch = wb[1]
            num_proc = min(word_batch['input_1'].shape[0], num_left)
            decoded_res = nt.decode_batch(self.test_func,word_batch['input_1'][0:num_proc],alphabet)
            for j in range(num_proc):
                source_str = nt.translate_array(out_batch['softmax_1'][j],alphabet)
                edit_dist = edistance.eval(decoded_res[j], source_str)
                #print(decoded_res[j] + ' | ' + source_str + ' = ' + str(edit_dist))
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(out_batch['softmax_1'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        #Save weights (new file)
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        #Save model   (overwrite)
        self.model.save(os.path.join(self.output_dir, 'model.h5'),overwrite=True)

class HistorySaver(keras.callbacks.History):
    '''
    Callback utilisé pour garder un historique des valeurs importantes (ex:'loss', 'accuracy')
        - Charge l'historique précédent et le complète si apprentissage repris
        - Sauvegarde à chaque epoch l'historique dans un fichier metrick.pk
    '''

    def __init__(self, init_epoch):
        self.init_epoch = init_epoch
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)

    def on_train_begin(self, logs=None):
        self.epoch = []
        if self.init_epoch >= 0:
            self.history = {}
        else:
            f = open(os.path.join(self.output_dir,"metrics.pk"),'rb')
            self.history = pickle.loads(f.read())
            for i in range (self.init_epoch):
                self.epoch.append(i)
            f.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        f = open(os.path.join(self.output_dir,"metrics.pk"),'wb')
        pickle.dump(self.history, f)
        
class TensorBoardWrapper(keras.callbacks.TensorBoard):
    '''
    Test pour utiliser tensorboard
    '''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(1):
            batch = self.batch_gen.get_batch(0, 200, train=False)
            imgs = batch[0]['the_input']
            tags = batch[0]['the_labels']
            imgl = batch[0]['input_length']
            tagl = batch[0]['label_length']
            ctct = [batch[1]['ctc']]
        self.validation_data = [imgs, tags,imgl,tagl,ctct,np.zeros(imgs.shape[0])]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)

def create_sparse(labels):
    indices = tf.where(tf.not_equal(labels, 0))
    values = tf.gather_nd(labels, indices)
    shape = tf.shape(labels, out_type=tf.int64)
    return tf.SparseTensor(indices, values, dense_shape=shape)


def train(run_name, img_w, img_h, start_epoch, stop_epoch, val_split, minibatch_size, max_str_len, max_samples, batch_memory_usage, type_model, use_ctc, **kwargs):
    """
    Train a model

    #Argument
        - run_name: String, the name of the folder used as save directory for metrics, model and weights.
        - start_epoch: Int, the starting epoch, 0 means start a new learning from scratch.
                    >0 means load weight for this epoch and continue the training from there.
        - stop_epoch: Int, the epoch limit, learning will stop once it reach this amount of epoch done.
        - img_w : Int, Width of the input images  (All images need to be normalized)
        - img_h : Int, Height of the input images (All images need to be normalized)

    """

    if K.image_data_format() == 'channels_first':
        print('NOT IMPLEMENTED !!!')

    channels = 1
    if channels==1 :
        input_shape = (img_w, img_h)
    else:
        input_shape = (img_w,img_h,channels)

    print('Build text image generator')
    #Generator used for training : load images, ground truth for training et validation set. See network_helper for more
    img_gen = TextImageGenerator(train_folder=datafolder_name,
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=8,
                                 val_split=val_split,
                                 alphabet=alphabet,
                                 absolute_max_string_len=max_str_len,
                                 max_samples=max_samples,
                                 acceptable_loss=10,
                                 memory_usage_limit=batch_memory_usage,
                                 channels=channels,
                                 use_ctc=use_ctc,
                                 noise="s&p")
    minibatch_size = img_gen.minibatch_size

    print('Building model...')
    #Current model
    model = None
    #Keras Backend Function used to obtain specific output from specific input, used in callbacks
    testfunc = None

    #Start from random initialization or used saved weights
    if start_epoch == 0:
        model, test_func = custom_model.get_model(type_model,input_shape,(max_str_len,len(alphabet)), img_gen, **kwargs)
    else:
        dir_path = os.path.join(OUTPUT_DIR,run_name)
        model, test_func = custom_model.get_model(type_model,input_shape,(max_str_len,len(alphabet)), img_gen, **kwargs)
        weight_file = os.path.join(dir_path,'weights%02d.h5' % (start_epoch-1))
        model.load_weights(weight_file, by_name=True)
        #est_func = K.function([model.get_layer('the_input').input], [model.get_layer('softmax').output])

    #Create and set all callbacks for training
    history = HistorySaver(start_epoch)
    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
    nan_cb = keras.callbacks.TerminateOnNaN()
    #tsboard = TensorBoardWrapper(img_gen, img_gen.get_val_steps(),log_dir='./logs',histogram_freq=1,write_grads=True, write_images=True)
    callbacks = [history, viz_cb, nan_cb]

    #Save model
    modelpath = "./data/output/"+run_name+"/model.h5"
    model.save(modelpath,overwrite=True)

    #Start training
    if (val_split > 0.0):
        hist = model.fit_generator(generator=img_gen.next_train(),
                            steps_per_epoch=img_gen.get_train_steps(),
                            epochs=stop_epoch,
                            validation_data=img_gen.next_val(),
                            validation_steps=img_gen.get_val_steps(),
                            callbacks=callbacks,
                            initial_epoch=start_epoch,
                            workers=12,
                            use_multiprocessing=True)
    else:
        hist = model.fit_generator(generator=img_gen.next_train(),
                            steps_per_epoch=img_gen.get_train_steps(),
                            epochs=stop_epoch,
                            callbacks=callbacks,
                            initial_epoch=start_epoch,
                            workers=12,
                            use_multiprocessing=True)
 
if __name__ == '__main__':
    if (len(sys.argv)<2):
        init_file = open('init.txt')
    else:
        init_file = open(sys.argv[1])
    init_content = init_file.readlines()

    for i in range(len(init_content)):
        init_content[i] = init_content[i].split('|')[1]
        init_content[i] = init_content[i].replace('\n','')


    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    OUTPUT_DIR = 'data/output/'
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
    i = 13
    if "Attention" in type_model and len(init_content) >= 16:
        c1 = [int(x) for x in init_content[13].split(',')]
        c2 = [int(x) for x in init_content[14].split(',')]
        enc = [int(x) for x in init_content[15].split(',')]
        dec = [int(x) for x in init_content[16].split(',')]
        kwargs = {'CNN' : [c1,c2],
                'Encoder' : enc,
                'Decoder' : dec}
        i = 17
        use_ctc = False
    else:
        kwargs = {}
        use_ctc = True
    str_optimizer = [x for x in init_content[i].split(',')]
    if len(str_optimizer)>1:
        if (str_optimizer[0]=='adam'):
            optarg = [0.001,0.9,0.999,0.00000001,0.0]
            for q in range(1,len(str_optimizer)):
                optarg[q-1] = float(str_optimizer[q])
            opt = keras.optimizers.Adam(optarg[0],optarg[1],optarg[2],optarg[3], optarg[4])
        if (str_optimizer[0]=='rmsprop'):
            optarg = [0.001,0.9,None,0.0]
            for q in range(1,len(str_optimizer)):
                optarg[q-1] = float(str_optimizer[q])
            opt = keras.optimizers.RMSprop(optarg[0],optarg[1],optarg[2],optarg[3])
        if (str_optimizer[0]=='adadelta'):
            optarg = [1.0,0.95,None,0.0]
            for q in range(1,len(str_optimizer)):
                optarg[q-1] = float(str_optimizer[q])
            opt = keras.optimizers.Adadelta(optarg[0],optarg[1],optarg[2],optarg[3])
        if (str_optimizer[0]=='SGD'):
            opt = keras.optimizers.Adam(float(optarg[0]),float(optarg[1]),float(optarg[2]))
    else:
        opt = str_optimizer[0]
    str_loss = init_content[i+1]
    kwargs["loss"]=keras.losses.get(str_loss)
    kwargs["opt"]=keras.optimizers.get(opt)
    train(run_name=run_name,start_epoch=start,stop_epoch=stop, type_model=type_model,
            img_w=image_width, img_h=image_height, val_split=val_split, minibatch_size=minibatch_size,
            max_str_len=max_str_len,max_samples=max_samples,batch_memory_usage=batch_memory_usage,use_ctc=use_ctc, **kwargs)
