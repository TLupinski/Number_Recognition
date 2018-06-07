import tensorflow
import keras
from keras import backend as K
from keras import regularizers
from keras.backend import tf
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, TimeDistributed, Bidirectional
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import network_helper as nt
from network_helper import TextImageGenerator, ctc_lambda_func
import seq2seq
from seq2seq import *
from seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
import recurrentshop
from recurrentshop import RecurrentSequential
from recurrentshop.engine import _OptionalInputPlaceHolder
import attention
from attention import Attention

'''
Defined custom models used in train_custom and test_custom
5 Models currently implemented :
CNN_RNN_CTC :       Basic model using a 2-Layer CNN followed by 2 Bidirectionnal GRU and a CTC decoder
ResCGRU:            Model using an Residual CNN followed by Bidirectionnal GRU layers based on [1]
VHLSTM              Model using a CNN followed by an Vertical LSTM then an Horizontal LSTM based on nothing.
AttentionBiLSTM :   Encoder-Decoder Attention Model using cells of Seq2Seq module for Keras
Attention:          Encoder-Decoder Attention Model using modified AttentionSeq2Seq from Seq2Seq module for keras
DisplayAttention:   Truncated Attention model used for visualization of intermediaire outputs.

#Argument needed:
    -input_shape : Shape of the inputs that will be fed to the model
    -output_shape: Shape of the output of the model (Needed only for attention model)
    -img_gen     : Generator of images / Can be replaced by two argument : number_of_classes and max_output_size
'''

def get_custom(type_model):
    if (type_model=='Attention' or type_model=='DisplayAttention'):
        return {'RecurrentSequential':RecurrentSequential,'_OptionalInputPlaceHolder':_OptionalInputPlaceHolder,
            'AltAttentionDecoderCell':AltAttentionDecoderCell, 'LSTMDecoderCell':LSTMDecoderCell}
    return {'ctc': lambda y_true, y_pred: y_pred}

def get_model(type_model, input_shape, output_shape, img_gen, weight_file=None):
    if (type_model=='Attention'):
        return Model_Attention(input_shape, output_shape, img_gen)
    if (type_model=='DisplayAttention'):
        return Model_DisplayAttention(input_shape, output_shape, img_gen,weight_file)
    if (type_model=='CNNRNNCTC'):
        return Model_CNN_RNN_CTC(input_shape, img_gen)
    if (type_model=='ResCGRUCTC'):
        return Model_ResCGRU(input_shape, img_gen)
    if (type_model=='VHLSTM'):
        return Model_VHLSTM(input_shape, img_gen)
    if (type_model=='DUMMY'):
        return Model_Dummy(input_shape, output_shape)
    return None,None
        
def Model_Attention(input_shape, output_shape, img_gen):
    model = ConvAttentionSeq2Seq(bidirectional=True,input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), dropout=0.25)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opt,metrics=['categorical_accuracy'])
    inp = model.get_layer('the_input')
    inputs = inp.input
    out = model.get_layer('the_output')
    y_pred = out.output
    test_func = K.function([inputs], [y_pred])
    return model, test_func

def Model_DisplayAttention(input_shape, output_shape, img_gen, weightfile):
    model = TruncConvAttentionSeq2Seq(bidirectional=True, input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), filename =weightfile)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy'])
    inp = model.get_layer('the_input')
    inputs = inp.input
    out = model.get_layer('the_output')
    y_pred = out.output[1]
    test_func = K.function([inputs], [y_pred])
    return model, test_func

def Model_CNN_RNN_CTC(input_shape, img_gen):
    # Input Parameters
    learning_rate=0.0001
    momentum1=0.9
    momentum2=0.999
    lambdad = 0.0001

    #Network Parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 50
    rnn_size = 512
    post_rnn_fcl_size = 100

    act = 'relu'

    img_w = input_shape[0]
    img_h = input_shape[1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    if (len(input_shape)<3):
        rs_shape = input_shape + (1,)
        input_data_rs = Reshape(target_shape=rs_shape)(input_data)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(input_data_rs)
    else:
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    rec_dropout = 0.2
    dropout = 0.25
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_merged = add([gru_1,gru_1b])
    gru_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)
    gru_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)

    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    print('Compiling model')
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    test_func = K.function([input_data], [y_pred])

    return model, test_func

def Model_ResCGRU(input_shape, img_gen):
    # Input Parameters
    learning_rate=0.0001
    momentum1=0.9
    lambdad = 0.0001

    # Network parameters
    kernel_size = (3, 3)
    time_dense_size = 50
    pool_size = 2
    rnn_size = 100
    post_rnn_fcl_size = 100
    act = tf.nn.relu
    img_w = input_shape[0]
    img_h = input_shape[1]
    
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    conv_1 = Conv2D(64, (5,5), padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    pool_1 =  MaxPooling2D(pool_size=(pool_size,pool_size),name='max1')(conv_1)
    conv_2a = Conv2D(64, (3,3), padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2a')(pool_1)
    conv_2b = Conv2D(64, (3,3), padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2b')(conv_2a)
    res_2d = add([pool_1,conv_2b])
    res_2  = Dropout(0.4)(res_2d)

    conv_3a = Conv2D(128, (3,3), strides=(1,2), activation=act,padding='same',kernel_initializer='he_normal',
                   name='conv3a')(res_2)
    conv_3b = Conv2D(128, (3,3), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv3b')(conv_3a)
    conv_3r = Conv2D(128, (1,1), strides=(1,2), padding='valid',kernel_initializer='he_normal',
                   name='conv3r')(res_2)
    res_3d= add([conv_3b,conv_3r])
    res_3 = Dropout(0.4)(res_3d)

    conv_4a = Conv2D(256, (3,3), strides=(1,2), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv4a')(res_3)
    conv_4b = Conv2D(256, (3,3), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv4b')(conv_4a)
    conv_4r = Conv2D(256, (1,1), strides=(1,2),padding='valid',kernel_initializer='he_normal',
                   name='conv4r')(res_3)
    res_4d = add([conv_4b,conv_4r])
    res_4  = Dropout(0.4)(res_4d)

    conv_5a = Conv2D(512, (3,3), strides=(1,2), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv5a')(res_4)
    conv_5b = Conv2D(512, (3,3), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv5b')(conv_5a)
    conv_5r = Conv2D(512, (1,1), strides=(1,2), padding='valid',kernel_initializer='he_normal',
                   name='conv5r')(res_4)
    res_5 = add([conv_5b,conv_5r])
    
    conv_to_rnn_dims = ((img_w // (2)), (1+(img_h // (16))) * 512)
    rnn_input = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(res_5)

    # cuts down input size going into RNN:
    #rnn_input = Dense(time_dense_size, activation=act, name='dense1')(conv_rs)

    rec_dropout = 0.2
    dropout = 0.25
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1  = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(rnn_input)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout)(rnn_input)
    gru_2  = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', recurrent_dropout=rec_dropout, dropout=dropout)(gru_1)
    gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2_b', recurrent_dropout=rec_dropout, dropout=dropout)(gru_1b)

    #One Fully Connected Layer before classification output
    inner_100  = Dense(post_rnn_fcl_size, activation="linear",use_bias=False,name='denseF1')(gru_2)
    inner_100b = Dense(post_rnn_fcl_size, activation="linear",use_bias=False,name='denseB1')(gru_2b)
    inner_class  = Dense(img_gen.get_output_size(), activation="linear",use_bias=False,name='denseF2')(inner_100)
    inner_classb = Dense(img_gen.get_output_size(), activation="linear",use_bias=False,name='denseB2')(inner_100b)
    inner_classb_reversed = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=2)) (inner_classb)

    # transforms RNN output to character activations:
    inner = add([inner_class,inner_classb_reversed])
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(nt.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    # clipnorm seems to speeds up convergence
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=momentum1, beta_2=momentum2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    return model, test_func

def Model_VHLSTM(input_shape, img_gen):
    # Input Parameters
    val_split = 0.1
    learning_rate=0.0001
    momentum1=0.9
    momentum2=0.999
    lambdad = 0.0001

    # Network parameters
    # 16
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 50
    rnn_v_size = 1
    rnn_h_size = 512
    post_rnn_fcl_size = 100
    act = tf.nn.relu

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    #Convolutional layers for features extraction :
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lambdad),
                   name='conv1')(input_data)
                   
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lambdad),
                   name='conv2')(inner)
                   
    maxpool2 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    print(maxpool2)

    # One layer of bidirectional Vertical GRUs


    # conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    # input_rnn = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(maxpool2)

    rec_dropout = 0.2
    dropout = 0.25

    # cuts down input size going into RNN:
    #input_rnn = Dense(time_dense_size, activation=act, name='dense1')(reshape)
    gru_h = []
    gru_vert = GRU(rnn_v_size, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambdad), recurrent_dropout=rec_dropout, dropout=dropout)
    for i in range(7):
        temp_input = Lambda(lambda x: x[:,i,:,:])(maxpool2)
        gru_h = gru_h + [gru_vert(temp_input)]
    input_rnn = concatenate(gru_h,axis=1)
    print(input_rnn)
    # Two layers of bidirectional Horizontal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_h_size, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambdad), name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(input_rnn)
    gru_1b = GRU(rnn_h_size, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambdad), name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout, go_backwards=True)(input_rnn)
    gru_1br = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)
    gru_merged = add([gru_1,gru_1br])
    gru_2 = GRU(rnn_h_size, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambdad), name='gru2', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)
    gru_2b = GRU(rnn_v_size, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambdad), name='gru2_b', recurrent_dropout=rec_dropout, dropout=dropout, go_backwards=True)(gru_merged)
    gru_2br = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)
    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2br]))
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(nt.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    # clipnorm seems to speeds up convergence
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=momentum1, beta_2=momentum2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    return model, test_func

def Model_AttentionBiLSTM(input_shape, img_gen):
    output_dim = 92
    output_length = 1
    hidden_dim = 7
    depth = [1,1]
    unroll = False
    stateful = False
    dropout = 0.2
    bidirectional = False
    if isinstance(depth, int):
        depth = (depth, depth)
    learning_rate=0.0001
    momentum1=0.9
    momentum2=0.999


    #Network Parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 50
    rnn_size = 512
    post_rnn_fcl_size = 100
    act = 'relu'
    img_w = input_shape[0]
    img_h = input_shape[1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    if len(input_shape) == 2:
        inputrs = Reshape(target_shape=(input_shape[0],input_shape[1],1))(input_data)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(inputrs)
    else :
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    rec_dropout = 0.2
    dropout = 0.25
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_merged = add([gru_1,gru_1b])
    gru_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)
    gru_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)
    gru_merged2 = add([gru_2,gru_2b])
    attention = Attention(RNN(rnn_size, return_sequences=True))(gru_merged2)

    model = Model(inputs=[_input], outputs=[attention])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=momentum1, beta_2=momentum2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function(inputs, [y_pred])
    return model, test_func

def Model_Dummy(input_shape, output_shape):
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    print(input_shape)
    rs = Reshape((4,-1))(input_data)
    out = Dense(11, activation='softmax', name='the_output')(rs)
    model = Model(inputs=input_data, outputs=out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss='mse', optimizer='adam')

    test_func = K.function([input_data], [out])

    return model, test_func
