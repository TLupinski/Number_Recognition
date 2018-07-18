from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential, Identity
from .cells import *
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, Reshape, Lambda, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.layers.recurrent import GRU, LSTM
from keras.layers.merge import add, concatenate
import keras.backend as K
import network_helper as nh
import numpy as np

#Loss function wrapper used to weight the loss output with a variable weight
def weighted_loss_init(loss_func, loss_weight):
    def weighted_loss(y_true,y_pred):
        return loss_func(y_true,y_pred) * loss_weight
    return weighted_loss
    
#Dummy function to pass the ctc loss computed before as the true loss
def ctc_loss(y_true, y_pred):
    return y_pred

#Dummy function to not count ctc decoded output into the loss calculation
def dum_loss(y_true,y_pred):
    return K.variable(0.0)

def cnn_init(CNN, inputrs, global_name=""):
    '''
    This function allows to create an CNN defined in an init_file.
    CNN is composed of two array:
        -First contains either -1 for MaxPooling2D layer or a positive integer X for a Conv2DLayer with X filters
        -Second array contains the kernel size k1,k2 for each layer
    inputrs must be of shape(Sample,Width,Height,Channel)
    global_name is a string applied before every layer name for differenciation.
    '''
    if not CNN==None:
        i = 1
        cpt_conv = 2
        cpt_pool = 1
        reduction = [1,1]
        n = CNN[0][0]
        k1,k2 = CNN[1][0:2]
        nb_filters = n
        cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name=global_name + 'conv1')(inputrs)
        while i < len(CNN[0]):
            n = CNN[0][i]
            k1,k2 = CNN[1][2*i:2*i+2]
            if (n > 0):
                cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name=global_name + 'conv'+str(cpt_conv))(cnn_inner)
                cnn_inner = BatchNormalization(name="bnp"+str(cpt_conv))(cnn_inner)
                cpt_conv = cpt_conv + 1
                nb_filters = n
            else:
                cnn_inner = MaxPooling2D(pool_size=(k1,k2),name=global_name + 'max'+str(cpt_pool))(cnn_inner)
                reduction[0] = reduction[0]*k1
                reduction[1] = reduction[1]*k2
                cpt_pool = cpt_pool + 1
            i = i+1
    else :
        conv_filters=32
        # First conv2D plus max pooling 2D
        conv1 = Conv2D(conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(inputrs)
        pool1 = MaxPooling2D(pool_size=(2,2), name='max1')(conv1)
        # Second conv2d plus max pooling 2D
        conv2 = Conv2D(2*conv_filters*2, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2,2), name='max2')(conv2)
        conv3 = Conv2D(3*conv_filters*4, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv3')(pool2)
        pool3 = MaxPooling2D(pool_size=(1,2), name='max3')(conv3)
        cnn_inner = Conv2D(4*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv4')(pool3)
    return cnn_inner, reduction, nb_filters

def patched_cnn_init(CNN, inputrs, global_name=""):
    '''
    This function allows to create an CNN defined in an init_file.
    CNN is composed of two array:
        -First contains either -1 for MaxPooling2D layer or a positive integer X for a Conv2DLayer with X filters
        -Second array contains the kernel size k1,k2 for each layer
    This function worked on a sequence on patch of an image instead of image directly
    inputrs must be of shape(Sample,Time,Width,Height,Channel)
    global_name is a string applied before every layer name for differenciation.
    '''
    if not CNN==None:
        i = 1
        cpt_conv = 2
        cpt_pool = 1
        reduction = [1,1]
        n = CNN[0][0]
        k1,k2 = CNN[1][0:2]
        nb_filters = n
        conv_layer = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name=global_name + 'conv1')
        cnn_inner = TimeDistributed(conv_layer)(inputrs)
        while i < len(CNN[0]):
            n = CNN[0][i]
            k1,k2 = CNN[1][2*i:2*i+2]
            if (n > 0):
                conv_layer = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name=global_name + 'conv'+str(cpt_conv))
                cnn_inner = TimeDistributed(conv_layer)(cnn_inner)
                cnn_inner = BatchNormalization(name="bnp"+str(cpt_conv))(cnn_inner)
                cpt_conv = cpt_conv + 1
                nb_filters = n
            else:
                max_layer = MaxPooling2D(pool_size=(k1,k2),name=global_name + 'max'+str(cpt_pool))
                cnn_inner = TimeDistributed(max_layer)(cnn_inner)
                reduction[0] = reduction[0]*k1
                reduction[1] = reduction[1]*k2
                cpt_pool = cpt_pool + 1
            i = i+1
    return cnn_inner, reduction, nb_filters

def encoder_init(input, postcshape, hidden_dim, depth, dropout=0, seq2seq=True, bidirectional=True, unroll=False, stateful=False, Encoder=None, global_name="", return_model = False):
    if Encoder==None:
        Encoder=[hidden_dim]*depth[0]
    else:
        if len(Encoder)<depth[0]:
            Encoder = Encoder + [hidden_dim]*(depth[0]-len(Encoder))
    encoder = RecurrentSequential(unroll=unroll, stateful=stateful, 
                                #   return_states=True, return_all_states=True, AllStateTransfer needs modification in the tensorflow backend
                                  return_sequences=True, name =global_name + 'encoder')
    encoder.add(LSTMCell(Encoder[0], batch_input_shape=postcshape[1:]))
    
    for k in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(Encoder[k]))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum', name=global_name + 'encoder')
        encoder.forward_layer.build(postcshape)
        encoder.backward_layer.build(postcshape)
        # patch
        encoder.layer = encoder.forward_layer
    if return_model:
        enc_input = Input(shape=postcshape[1:], name='encoder_input')
        encoded_out = encoder(enc_input)
        encoder_model = Model(inputs=[enc_input], outputs=[encoded_out])
        return encoder_model(input)
    return encoder(input)
    #State_transfer is not currently supported
    # if (state_transfer):
    #     encoded_outputs, _, encoder_states,_ ,_ = encoder(cnn_out)
    #     encoder_states._keras_shape = encoded_outputs._keras_shape
    #     encoded = concatenate([encoded_outputs,encoder_states], axis=1)
    # else :
    #     encoded = encoder(cnn_out)

def decoder_init(input, shape, input_dim, hidden_dim, output_dim, output_length, depth, dropout = 0, bidirectional=True, unroll=False, stateful=False, Decoder=None, AttentionCell=AltAttentionDecoderCell, global_name = "", return_model = False):
    if Decoder==None:
        Decoder=[hidden_dim]*depth[1]
    else:
        if len(Decoder)<depth[1]:
            Decoder = Decoder + [hidden_dim]*(depth[1]-len(Decoder))
    
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful, name='decoder')
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], input_dim)))
    if depth[1] == 1:
        decoder.add(AttentionCell)
    else:
        decoder.add(AttentionCell)
        for k in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=Decoder[k+1], hidden_dim=Decoder[k]))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=Decoder[-1]))
    return decoder(input)
    
def PatchedConvAttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True,
              unroll=False, stateful=False, dropout=0.0, state_transfer=False, 
              CNN=None, Encoder=None, Decoder=None, display_attention=False):
    '''
    This is an attention Seq2seq model with convolutionnal layers before for features extraction.
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.

    The  math:

            Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
            so H is a sequence of vectors of length m.

            Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:

    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j), alpha(i-1))
    alpha = softmax(energy)
    Where a is a feed forward network.

    '''
    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim
    input_shape = shape[1:] + (1,)
    img_w = input_shape[1]
    img_h = input_shape[2]
    dense_cnn_size = 1024
    _input = Input(batch_shape=shape, name='the_input')
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)
    global_name = "ord"
    #Adding all the CNN layers described in the init file or just a 3-layers CNN if nothing is given
    cnn_inner, reduction, nb_filters = patched_cnn_init(CNN, _inputrs, global_name)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = ((img_w // reduction[0]), (img_h // reduction[1]) * nb_filters)
    
    #RNN Encoder Part
    cnn_inner = Reshape((input_shape[0],-1))(cnn_inner)
    cnn_out = TimeDistributed(Dense(dense_cnn_size))(cnn_inner)
    postcshape = (shape[0],input_shape[0],dense_cnn_size)
    
    encoded = encoder_init(cnn_out, postcshape, hidden_dim, depth, dropout, True, bidirectional, unroll, stateful, Encoder)
    encoded_shape = (shape[0], shape[1], Encoder[-1])
    #Decoder Part
    AttentionCell = AltAttentionDecoderCellC(output_dim=output_dim, hidden_dim=Decoder[0])
    if display_attention:
        Decoder = [Decoder[0]]
        AttentionCell = AttentionDecoderCellDisplay(AttentionCell)
    decoded = decoder_init(encoded, encoded_shape, Encoder[-1], hidden_dim, output_dim, output_length, depth, dropout, bidirectional, unroll, stateful, Decoder, AttentionCell)

    inputs = [_input]
    if display_attention:
        output = Identity(name = 'the_output')(decoded)
    else:
        output = Softmax(name = 'the_output')(decoded)
    model = Model(inputs, output)
    return model

def TruncConvAttentionSeq2Seq(output_dim, output_length, filename, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True, unroll=False, stateful=False, dropout=0.0,
              CNN=None, Encoder=None, Decoder=None):
    '''
    Truncated ConvAttentionSeq2Seq model used to display the attention vector instead of the Decoder output.
    The Decoder uses a modified AttentionCell which output the attention vector and can only have one layer.

    '''
    state_transfer = False
    wmdl = ConvAttentionSeq2Seq(CNN=CNN, Encoder=Encoder, Decoder=Decoder, bidirectional=True,input_length=input_length, input_dim=input_dim, hidden_dim=hidden_dim, output_length=output_length, output_dim=output_dim, depth=(2,1))
    wmdl.load_weights(filename, by_name=True)
    print("Full Model Loaded")
    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim
    input_shape = (shape[1], shape[2], 1)
    conv_filters = 16
    pool_size = 2
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape, name='the_input')
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)

        #Adding all the CNN layers described in the init file or just a 3-layers CNN if nothing is given
    cnn_inner, reduction, nb_filters = patched_cnn_init(CNN, _inputrs, global_name)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = ((img_w // reduction[0]), (img_h // reduction[1]) * nb_filters)
    
    #RNN Encoder Part
    cnn_inner = Reshape((input_shape[0],-1))(cnn_inner)
    cnn_out = TimeDistributed(Dense(dense_cnn_size))(cnn_inner)
    postcshape = (shape[0],input_shape[0],dense_cnn_size)
    
    encoded = encoder_init(cnn_out, postcshape, hidden_dim, depth, dropout, True, bidirectional, unroll, stateful, Encoder)
    encoded_shape = (shape[0], shape[1], Encoder[-1])
    #Decoder Part
    AttentionCell = AttentionDecoderCellDisplay(AltAttentionDecoderCellC(output_dim=output_dim, hidden_dim=Decoder[0]))
    decoded = decoder_init(encoded, encoded_shape, Encoder[-1], hidden_dim, output_dim, output_length,
                           depth, dropout, bidirectional, unroll, stateful, Decoder[0], AttentionCell)

    #Load weights layer by layer because the global load_weights is not working for this.
    model = Model(inputs, out)
    for i in range(len(model.layers)):
        w = wmdl.get_layer(index=i).get_weights()
        if not w==None:
            model.get_layer(index=i).set_weights(w)
    return model

def ConvAttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True,
              unroll=False, stateful=False, dropout=0.0, state_transfer=False, 
              CNN=None, Encoder=None, Decoder=None, display_attention=False):
    '''
    This is an attention Seq2seq model with convolutionnal layers before for features extraction.
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.

    The  math:

            Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
            so H is a sequence of vectors of length m.

            Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:

    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j), alpha(i-1))
    alpha = softmax(energy)
    Where a is a feed forward network.

    '''
    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim

    input_shape = (32,shape[1]/32, shape[2], 1)
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape, name='the_input')
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)
    global_name = "ord"
    #Adding all the CNN layers described in the init file or just a 3-layers CNN if nothing is given
    cnn_inner, reduction, nb_filters = cnn_init(CNN, _inputrs, global_name)
    dense_cnn_size = 1024
    #RNN Encoder Part
    cnn_inner = Reshape((input_shape[0],-1))(cnn_inner)
    cnn_out = TimeDistributed(Dense(dense_cnn_size))(cnn_inner)
    postcshape = (shape[0],input_shape[0],dense_cnn_size)
    
    encoded = encoder_init(cnn_out, postcshape, hidden_dim, depth, dropout, True,
                           Bidirectional, unroll, stateful, Encoder, global_name)
    encoded_shape = (shape[0], shape[1], Encoder[-1])
    #Decoder Part
    AttentionCell = AltAttentionDecoderCell(output_dim=output_dim, hidden_dim=Decoder[0])
    if display_attention:
        Decoder = [Decoder[0]]
        AttentionCell = AttentionDecoderCellDisplay(AttentionCell)
    decoded = decoder_init(encoded, encoded_shape, Encoder[-1], hidden_dim, output_dim,
                           output_length, depth, dropout, bidirectional, unroll, 
                           stateful, Decoder, AttentionCell)

    inputs = [_input]
    if display_attention:
        output = Identity(name = 'the_output')(decoded)
    else:
        output = Softmax(name = 'the_output')(decoded)
    model = Model(inputs, output)
    model.summary()
    return model

class VLW(Callback):
    '''
        Callback used for dynamic loss weights variation during training
        At start, loss weight is [a,b] and it come to [a+c*d,b-c*d]
        for [AttentionLoss,CTC loss] resp.
    '''

    def __init__(self, a, b, c, d):
        self.alpha = a
        self.beta = b
        self.max_epoch = c
        self.delta = d

    def on_epoch_end(self, epoch, logs={}):
        if (epoch < self.max_epoch):
            K.set_value(self.alpha, K.get_value(self.alpha) + self.delta)
            K.set_value(self.beta, K.get_value(self.beta) - self.delta)

def ConvJointCTCAttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True,
              unroll=False, stateful=False, dropout=0.0, state_transfer=False, 
              CNN=None, Encoder=None, Decoder=None, glob_opt="adam", att_loss=keras.losses.mean_squared_error):
    '''
    This is the same network as the ConvAttetionSeq2Seq but with another branc added after the Encoder:
    One dense layer to convert Encoder output into probabilities and a CTC layer to decode the probabilities.

    The network hence has two output and two loss function:
        A loss for the attention branch which you can choose during initialization
        CTC_Loss for the CTC branch

    For weighting the loss function, a callback is used to transfer from [0.2,0.8] to [0.8,0.2]

    '''
    K.set_learning_phase(0)
    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim
    conv_filters = 16
    input_shape = (shape[1], shape[2], 1)
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape, name='the_input')
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)

    #Creation of the custom CNN describe in the init file or using a basic CNN
    global_name = "ord"
    cnn_inner, reduction, nb_filters = cnn_init(CNN, _inputrs, global_name)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = ((img_w // reduction[0]), (img_h // reduction[1]) * nb_filters)
    
    #RNN Encoder Part
    cnn_inner = Reshape((input_shape[0],-1))(cnn_inner)
    cnn_out = TimeDistributed(Dense(dense_cnn_size))(cnn_inner)
    postcshape = (shape[0],input_shape[0],dense_cnn_size)
    
    encoded = encoder_init(cnn_out, postcshape, hidden_dim, depth, dropout, True,
                           bidirectional, unroll, stateful, Encoder, return_model=True)
    encoded_shape = (shape[0], shape[1], Encoder[-1])
    #Decoder Part
    AttentionCell = AltAttentionDecoderCellD
    decoded = decoder_init(encoded, encoded_shape, Encoder[-1], hidden_dim, output_dim,
                           output_length, depth, dropout, bidirectional, unroll, stateful,
                           Decoder[0], AttentionCell, return_model=True)

    output_att = Softmax(name = 'the_output')(decoded)

    #CTC Part
        #Supplementary inputs needed for CTC
    labels = Input(name='the_labels', shape=[output_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    encoded_pred = Dense(output_dim, kernel_initializer='he_normal',
                  name='dense_ctc')(encoded)
    y_pred = Softmax(name='the_output_ctc')(encoded_pred)
    loss_ctc = Lambda(nh.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    output_ctc = y_pred#Lambda(nh.ctc_lambda_decode_func, output_shape=(1,), name='the_output_ctc')([y_pred, input_length])

    inputs = [_input, labels, input_length, label_length]
    model = Model(inputs, [output_att, loss_ctc, output_ctc])

    #Variables used for the dynamic loss weights
    alpha = K.variable(0.2)
    beta = K.variable(0.8)
    epoch = 15
    delta = 0.6/epoch

    jayer = model.get_layer(name='model_19')
    jayer.name = "ordencoder"
    jayer = model.get_layer(name='model_23')
    jayer.name = "decoder"
    
    mets = {'the_output':'categorical_accuracy'}
    model.compile(glob_opt, loss=[att_loss,ctc_loss,dum_loss], metrics = mets)#weighted_loss_init(att_loss,alpha),weighted_loss_init(ctc_loss,beta)
    model.summary()
    loss_cb = VLW(alpha,beta,epoch,delta)
    test_func = None
    return model, test_func, loss_cb

def ConvJointCTCFrozenAttentionSeq2Seq(filename, opt, loss, **kwargs):
    '''
        This is the same network as the ConvJointCTCAttentionSeq2Seq, but all layers are frozen except
        the Dense Layer before the CTC to use a already trained attention and train attention on it.

        However, the attention didn't used the RNN Encoder the same way the CTC does and this training failed
        because the CTC couldn't adapt itself without modifying the RNN Encoder.
    '''
    model, _ , _ = ConvJointCTCAttentionSeq2Seq(**kwargs)
    model.load_weights(filename,by_name=True)
    for i in range(len(model.layers)):
        print(i)
        layer = model.get_layer(index=i)
        layer.trainable = False
        if "ctc" in layer.name:
           layer.trainable = True
    mets = {'the_output':'categorical_accuracy'}
    model.compile(opt, loss=[loss, ctc_loss, dum_loss], loss_weights=[0.0,1.0,0.0], metrics = mets)
    model.summary()
    return model, None, None
