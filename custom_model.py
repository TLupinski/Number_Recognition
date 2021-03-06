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
from keras.optimizers import SGD, RMSprop
from keras.utils.data_utils import get_file
from keras.utils import plot_model
from keras.preprocessing import image
import network_helper as nt
from network_helper import TextImageGenerator, ctc_lambda_func
import seq2seq
from seq2seq import *
import recurrentshop
from recurrentshop import RecurrentSequential
from recurrentshop.engine import _OptionalInputPlaceHolder

'''
Defined custom models used in train_custom and test_custom
6 Models currently implemented :
CNNRNNCTC       :   Basic model using a 2-Layer CNN followed by 2 Bidirectionnal GRU and a CTC decoder
ResCNNRNNCTC    :   Basic model using a CNN with 2 Residual Block followed by 2 Bidirectionnal GRU and a CTC decoder
ResCGRU         :   Model using an Residual CNN followed by Bidirectionnal GRU layers described in [1].
ResCClassic     :   Model based on ResCGRU but the CNN is replaced by a classical ResidualCNN with residual blocks.
VHLSTM          :   Model using a CNN followed by an Vertical LSTM then an Horizontal LSTM
Attention       :   Encoder-Decoder Attention Model using modified AttentionSeq2Seq from Seq2Seq module for keras
PatchedAttention:   Encoder-Decoder Attention Model working with CNN applied to image patch instead of the complete image directly
DisplayAttention:   Truncated Attention model used for visualization of intermediaire outputs.
JointCTCAtt     :   Encoder-Decoder Attention Model with an additional CTC output after the Encoder.

#Argument needed:
    -input_shape : Shape of the inputs that will be fed to the model
    -output_shape: Shape of the output of the model (Needed only for attention model)
    -img_gen     : Generator of images / Can be replaced by two argument : number_of_classes and max_output_size

#Reference:
'''

def get_custom(type_model):
    if (type_model=='Attention' or type_model=='DisplayAttention'):
        return {'RecurrentSequential':RecurrentSequential,'_OptionalInputPlaceHolder':_OptionalInputPlaceHolder,
            'AltAttentionDecoderCell':AltAttentionDecoderCell, 'LSTMDecoderCell':LSTMDecoderCell}
    return {'ctc': lambda y_true, y_pred: y_pred}

def get_model(type_model, input_shape, output_shape, img_gen, weight_file=None, **kwargs):
    if (type_model=='Attention'):
        return Model_Attention(input_shape, output_shape, img_gen, **kwargs)
    if (type_model=='DisplayAttention'):
        return Model_DisplayAttention(input_shape, output_shape, img_gen, weight_file, **kwargs)
    if (type_model=='PatchedAttention'):
        return Model_PatchedAttention(input_shape, output_shape, img_gen, **kwargs)
    if (type_model=='CTCAttention'):
        return Model_CTCAttention(input_shape, output_shape, img_gen, **kwargs)
    if (type_model=='CTCFrozenAttention'):
        return Model_CTCFrozenAttention(input_shape, output_shape, img_gen, weight_file, **kwargs)
    if (type_model=='CNNRNNCTC'):
        return Model_CNN_RNN_CTC(input_shape, img_gen)
    if (type_model=='ResCNNRNNCTC'):
        return Model_ResCNN_RNN_CTC(input_shape, img_gen)
    if (type_model=='ResCGRUCTC'):
        return Model_ResCGRU(input_shape, img_gen)
    if (type_model=='ResCClassic'):
        return Model_ResCClasic(input_shape, img_gen)
    if (type_model=='VHLSTM'):
        return Model_VHLSTM(input_shape, img_gen)
    if (type_model=='DUMMY'):
        return Model_Dummy(input_shape, output_shape)
    return None,None
        
def Model_Attention(input_shape, output_shape, img_gen, loss='mse', opt='sgd', **kwargs):
    model = ConvAttentionSeq2Seq(bidirectional=True,input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), dropout=0.25, **kwargs)
    model.compile(loss=loss, optimizer=opt,metrics=['categorical_accuracy'])
    inp = model.get_layer('the_input')
    inputs = inp.input
    out = model.get_layer('the_output')
    y_pred = out.output
    test_func = K.function([inputs], [y_pred])
    return model, test_func, None

def Model_PatchedAttention(input_shape, output_shape, img_gen, loss='mse', opt='sgd', **kwargs):
    model = PatchedConvAttentionSeq2Seq(bidirectional=True,input_shape = input_shape, hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), dropout=0.25, **kwargs)
    model.compile(loss=loss, optimizer=opt,metrics=['categorical_accuracy'])
    inp = model.get_layer('the_input')
    inputs = inp.input
    out = model.get_layer('the_output')
    y_pred = out.output
    test_func = K.function([inputs], [y_pred])
    return model, test_func, None

def Model_CTCAttention(input_shape, output_shape, img_gen, loss, opt, **kwargs):
    model, test_func, callback = ConvJointCTCAttentionSeq2Seq(glob_opt = opt, att_loss = loss, bidirectional=True, input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), dropout=0.25, **kwargs)
    return model, test_func, callback

def Model_CTCFrozenAttention(input_shape, output_shape, img_gen, weight_file, **kwargs):
    model, test_func, callback = ConvJointCTCFrozenAttentionSeq2Seq("./data/output/FROZEN/FROZEN.h5", bidirectional=True, input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), dropout=0.25, **kwargs)
    return model, test_func, callback

def Model_DisplayAttention(input_shape, output_shape, img_gen, weightfile, loss, opt, **kwargs):
    model = TruncConvAttentionSeq2Seq(bidirectional=True, input_length=input_shape[0], input_dim=input_shape[1], hidden_dim=64, output_length=output_shape[0], output_dim=output_shape[1], depth=(2,1), filename =weightfile, **kwargs)
    model.compile(loss=loss, optimizer=opt,metrics=['categorical_accuracy'])
    inp = model.get_layer('the_input')
    inputs = inp.input
    out = model.get_layer('the_output')
    y_pred = out.output[1]
    test_func = K.function([inputs], [y_pred])
    return model, test_func, None

def Model_CNN_RNN_CTC(input_shape, img_gen):
    # Input Parameters
    learning_rate=0.0001
    momentum1=0.9
    momentum2=0.999
    lambdad = 0.0001

    #Network Parameters
    conv_filters = 64
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 64
    rnn_size = 64
    post_rnn_fcl_size = 100
    act = 'relu'
    rd=2
    
    if len(input_shape) == 3:
        img_w = input_shape[1]
        img_h = input_shape[2]
        use_patches = True
    else : 
        img_w = input_shape[0]
        img_h = input_shape[1]
        use_patches = False

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    if (len(input_shape)<4):
        rs_shape = input_shape + (1,)
        input_data_rs = Reshape(target_shape=rs_shape)(input_data)
        conv0 = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')
        if (use_patches):
            conv0 = TimeDistributed(conv0)
        inner = conv0(input_data_rs)
    else:
        conv0 = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')
        if (use_patches):
            conv0 = TimeDistributed(conv0)
        inner = conv0(input_data)
    maxpool0 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')
    if (use_patches):
        maxpool0 = TimeDistributed(maxpool0)
    inner = maxpool0(inner)
    conv_filters = conv_filters*2
    inner = td_res_connection(inner, rd, conv_filters, 3, keras.layers.ELU(alpha=1.0))
    # conv1 = Conv2D(conv_filters, kernel_size, padding='same',
    #                activation=act, kernel_initializer='he_normal',
    #                name='conv2')    
    # if (use_patches):
    #     conv1 = TimeDistributed(conv1)
    # inner = conv1(inner)
    maxpool1 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')
    if (use_patches):
        maxpool1 = TimeDistributed(maxpool1)
    inner = maxpool1(inner)
    conv_filters = conv_filters*2
    inner = td_res_connection(inner, rd, conv_filters, 3, keras.layers.ELU(alpha=1.0))

    inner = Reshape(target_shape=(input_shape[0],-1), name='reshape')(inner)

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
                  name='dense2', use_bias=False)(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='the_output')(inner)
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    decode_length = Input(name='input_length_decode', batch_shape=[None], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    test_func = K.function([input_data],[y_pred])

    print('Compiling model')
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    top_path=5
    top_k_decoded = K.ctc_decode(y_pred, decode_length, greedy=False,beam_width=20,top_paths=top_path)
    outout = [y_pred]
    for i in range (top_path):
        outout = outout+[top_k_decoded[0][i]]
    decoder = K.function([input_data, decode_length], outout)

    return model, decoder, None

def Model_ResCNN_RNN_CTC(input_shape, img_gen):
    # Input Parameters
    learning_rate=0.0001
    momentum1=0.9
    momentum2=0.999
    lambdad = 0.0001

    #Network Parameters
    conv_filters = 64
    kernel_size = (5, 5)
    pool_size = 3
    time_dense_size = 256
    rnn_size = 128
    post_rnn_fcl_size = 100
    rd = 2
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
                    activation=act, kernel_initializer='he_normal', name='conv1')(input_data)
    
    mp_0 = MaxPooling2D(pool_size=(2,2),name="max1")(inner)
    res_1 = td_res_connection(mp_0, rd, 64, 3, act)
    mp_1 = MaxPooling2D(pool_size=(2,2),name="max1")(res_1)
    res_2 = td_res_connection(mp_1, rd, 128, 3, act)
    mp_2 = MaxPooling2D(pool_size=(2,2),name="max2")(res_2)

    conv_to_rnn_dims = ((img_w // (8)), ((img_h // (8))) * 128)
    rnn_input = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(mp_2)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(rnn_input)

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
                  name='dense2', use_bias=False)(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='the_output')(inner)
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    test_func = K.function([input_data],[y_pred])

    print('Compiling model')
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
    return model, test_func, None

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
    act = 'relu'
    img_w = input_shape[0]
    img_h = input_shape[1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    input_data_rs = Reshape(target_shape=input_shape+(1,))(input_data)
    conv_1 = Conv2D(64, (5,5), padding='same', activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data_rs)
    conv_1 = BatchNormalization()(conv_1)
    # conv_1 = Dropout(0.4)(conv_1)
    pool_1 =  MaxPooling2D(pool_size=(pool_size,pool_size),name='max1')(conv_1)
    conv_2a = Conv2D(64, (3,3), padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2a')(pool_1)
    conv_2a = BatchNormalization()(conv_2a)
    conv_2b = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal',
                   name='conv2b')(conv_2a)
    conv_2b = BatchNormalization()(conv_2b)
    res_2 = add([pool_1,conv_2b])
    # res_2  = Dropout(0.4)(res_2)

    conv_3a = Conv2D(128, (3,3), strides=(2,2), activation=act,padding='same',kernel_initializer='he_normal',
                   name='conv3a')(res_2)
    conv_3a = BatchNormalization()(conv_3a)
    conv_3b = Conv2D(128, (3,3), padding='same',kernel_initializer='he_normal',
                   name='conv3b')(conv_3a)
    conv_3b = BatchNormalization()(conv_3b)
    conv_3r = Conv2D(128, (1,1), strides=(2,2), padding='same',
                   name='conv3r')(res_2)
    conv_3r = BatchNormalization()(conv_3r)
    res_3= add([conv_3b,conv_3r])
    # res_3 = Dropout(0.4)(res_3)

    conv_4a = Conv2D(256, (3,3), strides=(2,2), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv4a')(res_3)
    conv_4a = BatchNormalization()(conv_4a)
    conv_4b = Conv2D(256, (3,3), padding='same',kernel_initializer='he_normal',
                   name='conv4b')(conv_4a)
    conv_4b = BatchNormalization()(conv_4b)
    conv_4r = Conv2D(256, (1,1), strides=(2,2),padding='same',
                   name='conv4r')(res_3)
    conv_4r = BatchNormalization()(conv_4r)
    res_4 = add([conv_4b,conv_4r])
    # res_4  = Dropout(0.4)(res_4)

    conv_5a = Conv2D(512, (3,3), strides=(1,2), activation=act, padding='same',kernel_initializer='he_normal',
                   name='conv5a')(res_4)
    conv_5a = BatchNormalization()(conv_5a)
    conv_5b = Conv2D(512, (3,3), padding='same',kernel_initializer='he_normal',
                   name='conv5b')(conv_5a)
    conv_5b = BatchNormalization()(conv_5b)
    conv_5r = Conv2D(512, (1,1), strides=(1,2), padding='same',
                   name='conv5r')(res_4)
    conv_5r = BatchNormalization()(conv_5r)
    res_5 = add([conv_5b,conv_5r])
    conv_to_rnn_dims = ((img_w // (8)), ((img_h // (16))) * 512)
    rnn_input = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(res_5)

    # cuts down input size going into RNN:
    #rnn_input = Dense(time_dense_size, activation=act, name='dense1')(conv_rs)

    rec_dropout = 0.25
    dropout = 0.5
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
    y_pred = Activation('softmax', name='the_output')(inner)

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
    opt = keras.optimizers.Adadelta()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    return model, test_func, None

def res_connection(i, residual_depth, n_filters, k_size, activation="relu"):
    from keras.layers import Conv2D, Activation, add
    x = Conv2D(n_filters, (k_size,k_size), padding="same")(i)
    orig_x = x
    x = Activation(activation)(x)
    for aRes in range(0, residual_depth):
        if aRes < residual_depth-1:
            x = Conv2D(n_filters, (k_size,k_size), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)
        else:
            x = Conv2D(n_filters, (k_size,k_size), padding="same")(x)
            x = BatchNormalization()(x)
    x = add([orig_x, x])
    x = Activation(activation)(x)
    return x

def td_res_connection(i, residual_depth, n_filters, k_size, activation=Activation("relu")):
    from keras.layers import Conv2D, Activation, add
    x = TimeDistributed(Conv2D(n_filters, (k_size,k_size), padding="same"))(i)
    #x = BatchNormalization()(x)
    orig_x = x
    x = activation(x)
    for aRes in range(0, residual_depth):
        if aRes < residual_depth-1:
            x = TimeDistributed(Conv2D(n_filters, (k_size,k_size), padding="same"))(x)
            x = BatchNormalization()(x)
            x = activation(x)
        else:
            x = TimeDistributed(Conv2D(n_filters, (k_size,k_size), padding="same"))(x)
            x = BatchNormalization()(x)
    x = add([orig_x, x])
    #x = activation(x)
    return x

def Model_ResCClasic(input_shape, img_gen):
    # Input Parameters
    learning_rate=0.0001
    momentum1=0.9
    lambdad = 0.0001

    K.set_learning_phase(1)
    # Network parameters
    kernel_size = (3, 3)
    time_dense_size =1024
    pool_size = 2
    rnn_size = 128
    post_rnn_fcl_size = 100
    act = keras.layers.ELU(alpha=1.0)
    img_w = input_shape[0]
    img_h = input_shape[1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    input_data_rs = Reshape(target_shape=input_shape+(1,))(input_data)
    rd = 2
    
    conv_filters=64
    conv_1 = TimeDistributed(Conv2D(conv_filters, (5,5), padding='same', activation=act, kernel_initializer='he_normal', name='conv1'))(input_data_rs)
    mp_0 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1), padding='same'))(conv_1)
    res_1 = td_res_connection(mp_0, rd, conv_filters, 3, act)
    mp_1 = TimeDistributed(MaxPooling2D(pool_size=(2,2),name="max1"))(res_1)
    conv_filters = conv_filters*2
    res_2 = td_res_connection(mp_1, rd, conv_filters, 3, act)
    mp_2 = TimeDistributed(MaxPooling2D(pool_size=(2,2),name="max2"))(res_2)
    conv_filters = conv_filters*2
    res_3 = td_res_connection(mp_2, rd, conv_filters, 3, act)
    mp_3 = TimeDistributed(MaxPooling2D(pool_size=(2,2),name="max3"))(res_3)
    conv_filters = conv_filters*2
    res_4 = td_res_connection(mp_3, rd, conv_filters, 3, act)

    cnn_inner = Reshape((input_shape[0],-1))(mp_3)
    cnn_out = TimeDistributed(Dense(time_dense_size))(cnn_inner)
    # cuts down input size going into RNN:

    rec_dropout = 0.2
    dropout = 0.25
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(cnn_out)
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout)(cnn_out)
    gru_merged = add([gru_1,gru_1b])
    gru_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)
    gru_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b', recurrent_dropout=rec_dropout, dropout=dropout)(gru_merged)

    # transforms RNN output to character activations:
    rnnout = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2', use_bias=False)(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='the_output')(rnnout)

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
    opt = keras.optimizers.Adadelta()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    model.summary()
    return model, test_func, None

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

#Dummy function to pass the ctc loss computed before as the true loss
def ctc_loss(y_true, y_pred):
    return y_pred

#Dummy function to not count ctc decoded output into the loss calculation
def dum_loss(y_true,y_pred):
    return K.variable(0.0)

def Model_Dummy(input_shape, output_shape):
    '''
        Model used for testing
    '''
    # model = yolo(input_shape, output_shape)
    # model.summary()
    # for i in range(len(model.layers)):
    #     layer = model.get_layer(index=i)
    #     layer.trainable = True
    # model.compile(loss=['mse','mse'], loss_weights=[1.0,1.0], optimizer='adam')
    # model.summary()
    K.set_learning_phase(0)
    conv_filters=16
    kernel_size=3
    pool_size=(2,2)
    rnn_size=64
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    rs_shape = input_shape + (1,)
    input_data_rs = Reshape(target_shape=rs_shape)(input_data)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(input_data_rs)
    inner = MaxPooling2D(pool_size=pool_size, name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=pool_size, name='max2')(inner)

    conv_to_rnn_dims = (256 // (pool_size[0] ** 2), (32 // (pool_size[1] ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)


    rec_dropout = 0.2
    dropout = 0.25
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b', recurrent_dropout=rec_dropout, dropout=dropout)(inner)
    gru_merged = add([gru_1,gru_1b])
    y_pred = Dense(11, activation='softmax', name='the_output')(gru_merged)

    labels = Input(name='the_labels', shape=[9], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    decode_length = Input(name='input_length_decode', batch_shape=[None], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(nt.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data,labels,input_length,label_length,decode_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    opt = RMSprop(0.0001)
    model.compile(loss=ctc_loss, optimizer=opt)
    
    top_path=5
    top_k_decoded = K.ctc_decode(y_pred, decode_length, greedy=False,beam_width=20,top_paths=top_path)
    outout = [y_pred]
    for i in range (top_path):
        outout = outout+[top_k_decoded[0][i]]
    decoder = K.function([input_data, decode_length], outout)

    return model, decoder, None