from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential, Identity
from .cells import Softmax, LSTMDecoderCell, AttentionDecoderCell, AltAttentionDecoderCell, AltAttentionDecoderCellD
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, Reshape, Lambda, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


def SimpleSeq2Seq(output_dim, output_length, hidden_dim=None, input_shape=None,
                  batch_size=None, batch_input_shape=None, input_dim=None,
                  input_length=None, depth=1, dropout=0.0, unroll=False,
                  stateful=False):

    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decodes the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence
    elements. The input sequence and output sequence may differ in length.

    Arguments:

    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
            there will be 3 LSTMs on the enoding side and 3 LSTMs on the
            decoding side. You can also specify depth as a tuple. For example,
            if depth = (4, 5), 4 LSTMs will be added to the encoding side and
            5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.

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
    encoder = RecurrentSequential(unroll=unroll, stateful=stateful)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[-1])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    decoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  decode=True, output_length=output_length)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], hidden_dim)))

    if depth[1] == 1:
        decoder.add(LSTMCell(output_dim))
    else:
        decoder.add(LSTMCell(hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMCell(hidden_dim))
    decoder.add(Dropout(dropout))
    decoder.add(LSTMCell(output_dim))

    _input = Input(batch_shape=shape)
    x = encoder(_input)
    output = decoder(x)
    return Model(_input, output)

def Seq2Seq(output_dim, output_length, batch_input_shape=None,
            input_shape=None, batch_size=None, input_dim=None, input_length=None,
            hidden_dim=None, depth=1, broadcast_state=True, unroll=False,
            stateful=False, inner_broadcast_state=True, teacher_force=False,
            peek=False, dropout=0.):

    '''
    Seq2seq model based on [1] and [2].
    This model has the ability to transfer the encoder hidden state to the decoder's
    hidden state(specified by the broadcast_state argument). Also, in deep models
    (depth > 1), the hidden state is propogated throughout the LSTM stack(specified by
    the inner_broadcast_state argument. You can switch between [1] based model and [2]
    based model using the peek argument.(peek = True for [2], peek = False for [1]).
    When peek = True, the decoder gets a 'peek' at the context vector at every timestep.

    [1] based model:

            Encoder:
            X = Input sequence
            C = LSTM(X); The context vector

            Decoder:
    y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
    y(0) = LSTM(s0, C); C is the context vector from the encoder.

    [2] based model:

            Encoder:
            X = Input sequence
            C = LSTM(X); The context vector

            Decoder:
    y(t) = LSTM(s(t-1), y(t-1), C)
    y(0) = LSTM(s0, C, C)
    Where s is the hidden state of the LSTM (h and c), and C is the context vector
    from the encoder.

    Arguments:

    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
                    there will be 3 LSTMs on the enoding side and 3 LSTMs on the
                    decoding side. You can also specify depth as a tuple. For example,
                    if depth = (4, 5), 4 LSTMs will be added to the encoding side and
                    5 LSTMs will be added to the decoding side.
    broadcast_state : Specifies whether the hidden state from encoder should be
                                      transfered to the deocder.
    inner_broadcast_state : Specifies whether hidden states should be propogated
                                                    throughout the LSTM stack in deep models.
    peek : Specifies if the decoder should be able to peek at the context vector
               at every timestep.
    dropout : Dropout probability in between layers.


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

    encoder = RecurrentSequential(readout=True, state_sync=inner_broadcast_state,
                                  unroll=unroll, stateful=stateful,
                                  return_states=broadcast_state)
    for _ in range(depth[0]):
        encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], hidden_dim)))
        encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(hidden_dim))
    dense1.supports_masking = True
    dense2 = Dense(output_dim)

    decoder = RecurrentSequential(readout='add' if peek else 'readout_only',
                                  state_sync=inner_broadcast_state, decode=True,
                                  output_length=output_length, unroll=unroll,
                                  stateful=stateful, teacher_force=teacher_force)

    for _ in range(depth[1]):
        decoder.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
                                    batch_input_shape=(shape[0], output_dim)))

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True
    encoded_seq = dense1(_input)
    encoded_seq = encoder(encoded_seq)
    if broadcast_state:
        assert type(encoded_seq) is list
        states = encoded_seq[-2:]
        encoded_seq = encoded_seq[0]
    else:
        states = None
    encoded_seq = dense2(encoded_seq)
    inputs = [_input]
    if teacher_force:
        truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
        truth_tensor._keras_history[0].supports_masking = True
        inputs += [truth_tensor]


    decoded_seq = decoder(encoded_seq,
                          ground_truth=inputs[1] if teacher_force else None,
                          initial_readout=encoded_seq, initial_state=states)
    
    model = Model(inputs, decoded_seq)
    model.encoder = encoder
    model.decoder = decoder
    return model

def convSeq2Seq(output_dim, output_length, batch_input_shape=None,
            input_shape=None, batch_size=None, input_dim=None, input_length=None,
            hidden_dim=None, depth=1, broadcast_state=True, unroll=False,
            stateful=False, inner_broadcast_state=True, teacher_force=True,
            peek=True, dropout=0.):
    '''
    This is an attention Seq2seq model based on [3] with convolutionnal layers before for features extraction.
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
    energy = a(s(i-1), H(j))
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
    input_shape = (shape[1], shape[2], 1)
    conv_filters = 16
    kernel_size = 3
    pool_size = 2
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)
    # First conv2D plus max pooling 2D
    inner = Conv2D(conv_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    # Second conv2D plus max pooling 2D
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)

    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    cnn_out = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    postcshape = (shape[0],conv_to_rnn_dims[0],conv_to_rnn_dims[1])
    
    encoder = RecurrentSequential(readout=True, state_sync=inner_broadcast_state,
                                  unroll=unroll, stateful=stateful,
                                  return_states=broadcast_state)
    for _ in range(depth[0]):
        encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], hidden_dim)))
        encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(hidden_dim))
    dense1.supports_masking = True
    dense2 = Dense(output_dim)

    decoder = RecurrentSequential(readout='add' if peek else 'readout_only',
                                  state_sync=inner_broadcast_state, decode=True,
                                  output_length=output_length, unroll=unroll,
                                  stateful=stateful, teacher_force=teacher_force, name='softmax_1')

    for _ in range(depth[1]):
        decoder.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
                                    batch_input_shape=(shape[0], output_dim)))

    encoded_seq = dense1(cnn_out)
    encoded_seq = encoder(encoded_seq)
    if broadcast_state:
        assert type(encoded_seq) is list
        states = encoded_seq[-2:]
        encoded_seq = encoded_seq[0]
    else:
        states = None
    encoded_seq = dense2(encoded_seq)
    inputs = [_input]
    if teacher_force:
        truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
        truth_tensor._keras_history[0].supports_masking = True
        inputs += [truth_tensor]


    decoded_seq = decoder(encoded_seq,
                          ground_truth=inputs[1] if teacher_force else None,
                          initial_readout=encoded_seq, initial_state=states)
    
    model = Model(inputs, decoded_seq)
    return model

def AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
                     batch_size=None, input_shape=None, input_length=None,
                     input_dim=None, hidden_dim=None, depth=1,
                     bidirectional=True, unroll=False, stateful=False, dropout=0.0):
    '''
    This is an attention Seq2seq model based on [3].
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
    energy = a(s(i-1), H(j))
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

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True

    encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
        encoder.forward_layer.build(shape)
        encoder.backward_layer.build(shape)
        # patch
        encoder.layer = encoder.forward_layer

    encoded = encoder(_input)
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    
    inputs = [_input]
    decoded = decoder(encoded)
    output = Softmax(axis=-1,name = 'softmax_1')(decoded)
    model = Model(inputs, output)
    return model

def SimpleConvAttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True, unroll=False, stateful=False, dropout=0.0):
    '''
    This is an attention Seq2seq model based on [3] with convolutionnal layers before for features extraction.
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
    energy = a(s(i-1), H(j))
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
    conv_fiter=16
    kernel_size = (3,3)
    input_shape = (shape[1], shape[2], 1)
    pool_size = 2
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)
    # First conv2D plus max pooling 2D
    inner = Conv2D(conv_filter, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    # First conv2D plus max pooling 2D
    inner = Conv2D(conv_filter, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    # Reshape to correct rnn inputs
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filter)
    cnn_out = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    postcshape = (shape[0],conv_to_rnn_dims[0],conv_to_rnn_dims[1])
    encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], conv_to_rnn_dims[1])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum', name='encoder')
        encoder.forward_layer.build(postcshape)
        encoder.backward_layer.build(postcshape)
        # patch
        encoder.layer = encoder.forward_layer

    encoded = encoder(cnn_out)
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful, name='decoder')
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        attention_cell = AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim)
        decoder.add(attention_cell)
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    
    inputs = [_input]
    decoded = decoder(encoded)
    output = Softmax(axis=-1,name = 'softmax_1')(decoded)
    model = Model(inputs, [output])
    return model

def TruncConvAttentionSeq2Seq(output_dim, output_length, filename, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True, unroll=False, stateful=False, dropout=0.0):

    wmdl = ConvAttentionSeq2Seq(bidirectional=True,input_length=input_length, input_dim=input_dim, hidden_dim=hidden_dim, output_length=output_length, output_dim=output_dim, depth=(2,1))
    wmdl.load_weights(filename)
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

    _inputrs = Reshape(target_shape=input_shape)(_input)
    if not CNN==None:
        i = 1
        cpt_conv = 2
        cpt_pool = 1
        reduction = [1,1]
        n = CNN[0][0]
        k1,k2 = CNN[1][0:2]
        nb_filters = n
        cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
        print(cnn_inner)
        while i < len(CNN[0]):
            n = CNN[0][i]
            k1,k2 = CNN[1][2*i:2*i+2]
            if (n > 0):
                cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name='conv'+str(cpt_conv))(cnn_inner)
                cpt_conv = cpt_conv + 1
                nb_filters = n
            else:
                cnn_inner = MaxPooling2D(pool_size=(k1,k2),name='max'+str(cpt_pool))(cnn_inner)
                reduction[0] = reduction[0]*k1
                reduction[1] = reduction[1]*k2
                cpt_pool = cpt_pool + 1
            print(cnn_inner)
            i = i+1
    else :
        # First conv2D plus max pooling 2D
        conv1 = Conv2D(conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
        pool1 = MaxPooling2D(pool_size=(2,2), name='max1')(conv1)
        # Second conv2d plus max pooling 2D
        conv2 = Conv2D(2*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2,2), name='max2')(conv2)
        conv3 = Conv2D(3*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv3')(pool2)
        pool3 = MaxPooling2D(pool_size=(1,2), name='max3')(conv3)
        cnn_inner = Conv2D(4*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv4')(pool3)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = ((img_w // reduction[0]), (img_h // reduction[1]) * nb_filters)
    cnn_out = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(cnn_inner)
    postcshape = (shape[0],conv_to_rnn_dims[0],conv_to_rnn_dims[1])
    if Encoder==None:
        Encoder=[hidden_dim]*depth[0]
    else:
        if len(Encoder)<depth[0]:
            Encoder = Encoder + [hidden_dim]*(depth[0]-len(Encoder))

    encoder = RecurrentSequential(unroll=True, stateful=stateful, 
                                #   return_states=True, return_all_states=True,
                                  return_sequences=True, name ='encoder')
    encoder.add(LSTMCell(Encoder[0], batch_input_shape=(shape[0], conv_to_rnn_dims[1])))

    for k in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(Encoder[k]))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum', name='encoder')
        encoder.forward_layer.build(postcshape)
        encoder.backward_layer.build(postcshape)
        # patch
        encoder.layer = encoder.forward_layer

    #encoded = encoder(cnn_out)
    if (state_transfer):
        encoded_outputs, _, encoder_states,_ ,_ = encoder(cnn_out)
        encoder_states._keras_shape = encoded_outputs._keras_shape
        encoded = concatenate([encoded_outputs,encoder_states], axis=1)
    else :
        encoded = encoder(cnn_out)

    if Decoder==None:
        Decoder=[hidden_dim]*depth[1]
    else:
        if len(Decoder)<depth[1]:
            Decoder = Decoder + [hidden_dim]*(depth[1]-len(Decoder))
    
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful, name='decoder')
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], Encoder[-1])))
    decoder.add(AltAttentionDecoderCellD(output_dim=output_dim, hidden_dim=Decoder[0]))
    inputs = [_input]
    out = decoder(encoded)
    model = Model(inputs, out)
    for i in range(len(model.layers)):
        w = wmdl.get_layer(index=i).get_weights()
        if not w==None:
            model.get_layer(index=i).set_weights(w)
    model.get_layer(name='the_output').get_cell(name='attention_decoder_cell_1').set_weights(wmdl.get_layer(name='decoder').get_cell(name='attention_decoder_cell_1').get_weights())
    return model

def ConvAttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
              batch_size=None, input_shape=None, input_length=None,
              input_dim=None, hidden_dim=None, depth=1, bidirectional=True,
              unroll=False, stateful=False, dropout=0.0, state_transfer=False, 
              CNN=None, Encoder=None, Decoder=None):
    '''
    This is an attention Seq2seq model based on [3] with convolutionnal layers before for features extraction.
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
    energy = a(s(i-1), H(j))
    alpha = softmax(energy)
    Where a is a feed forward network.

    '''
    K.set_learning_phase(1)
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
    pool_size = 2
    input_shape = (shape[1], shape[2], 1)
    img_w = input_shape[0]
    img_h = input_shape[1]
    _input = Input(batch_shape=shape, name='the_input')
    _input._keras_history[0].supports_masking = True
    #Reshaping input for convlayer
    _inputrs = Reshape(target_shape=input_shape)(_input)
    if not CNN==None:
        i = 1
        cpt_conv = 2
        cpt_pool = 1
        reduction = [1,1]
        n = CNN[0][0]
        k1,k2 = CNN[1][0:2]
        nb_filters = n
        cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
        while i < len(CNN[0]):
            n = CNN[0][i]
            k1,k2 = CNN[1][2*i:2*i+2]
            if (n > 0):
                cnn_inner = Conv2D(n, (k1,k2), padding='same', activation='relu', kernel_initializer='he_normal', name='conv'+str(cpt_conv))(cnn_inner)
                cpt_conv = cpt_conv + 1
                nb_filters = n
            else:
                cnn_inner = MaxPooling2D(pool_size=(k1,k2),name='max'+str(cpt_pool))(cnn_inner)
                reduction[0] = reduction[0]*k1
                reduction[1] = reduction[1]*k2
                cpt_pool = cpt_pool + 1
            i = i+1
    else :
        # First conv2D plus max pooling 2D
        conv1 = Conv2D(conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(_inputrs)
        pool1 = MaxPooling2D(pool_size=(2,2), name='max1')(conv1)
        # Second conv2d plus max pooling 2D
        conv2 = Conv2D(2*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2,2), name='max2')(conv2)
        conv3 = Conv2D(3*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv3')(pool2)
        pool3 = MaxPooling2D(pool_size=(1,2), name='max3')(conv3)
        cnn_inner = Conv2D(4*conv_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv4')(pool3)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = ((img_w // reduction[0]), (img_h // reduction[1]) * nb_filters)
    cnn_out = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(cnn_inner)
    postcshape = (shape[0],conv_to_rnn_dims[0],conv_to_rnn_dims[1])
    if Encoder==None:
        Encoder=[hidden_dim]*depth[0]
    else:
        if len(Encoder)<depth[0]:
            Encoder = Encoder + [hidden_dim]*(depth[0]-len(Encoder))

    encoder = RecurrentSequential(unroll=True, stateful=stateful, 
                                #   return_states=True, return_all_states=True,
                                  return_sequences=True, name ='encoder')
    encoder.add(LSTMCell(Encoder[0], batch_input_shape=(shape[0], conv_to_rnn_dims[1])))

    for k in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(Encoder[k]))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum', name='encoder')
        encoder.forward_layer.build(postcshape)
        encoder.backward_layer.build(postcshape)
        # patch
        encoder.layer = encoder.forward_layer

    #encoded = encoder(cnn_out)
    if (state_transfer):
        encoded_outputs, _, encoder_states,_ ,_ = encoder(cnn_out)
        encoder_states._keras_shape = encoded_outputs._keras_shape
        encoded = concatenate([encoded_outputs,encoder_states], axis=1)
    else :
        encoded = encoder(cnn_out)

    if Decoder==None:
        Decoder=[hidden_dim]*depth[1]
    else:
        if len(Decoder)<depth[1]:
            Decoder = Decoder + [hidden_dim]*(depth[1]-len(Decoder))
    
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful, name='decoder')
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], Encoder[-1])))
    if depth[1] == 1:
        decoder.add(AltAttentionDecoderCell(output_dim=output_dim, hidden_dim=Decoder[0]))
    else:
        decoder.add(AltAttentionDecoderCell(output_dim=output_dim, hidden_dim=Decoder[0]))
        for k in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=Decoder[k+1], hidden_dim=Decoder[k]))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=Decoder[-1]))
    
    inputs = [_input]
    decoded = decoder(encoded)
    output = Softmax(name = 'the_output')(decoded)
    model = Model(inputs, output)
    model.summary()
    return model