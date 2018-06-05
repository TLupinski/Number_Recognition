from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM


def get_model(time_dense_size, img_w, img_h, channels, max_string_len, vocab_size, conv_filters, kernel_size, pool_size, rnn_size, callback_func, activation='relu'):

    # Define the input shape
    input_shape = (img_w, img_h, channels)

    print 'input_shape {}'.format(input_shape)
    print 'kernel_size {}'.format(kernel_size)
    print 'num conv filter {}'.format(conv_filters)
    print 'Activation {} '.format(activation)

    # The input data of shape input_shape
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    # First conv2D plus max pooling 2D
    inner = Conv2D(conv_filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    # Second conv2D plus max pooling 2D
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=activation, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    # Reshape to correct rnn inputs
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=activation, name='dense1')(inner)

    # Two layers of bidirectional LSTM
    lstm_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    lstm1_merged = add([lstm_1, lstm_1b])
    lstm_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)

    # transforms RNN output to character activations:
    inner = Dense(vocab_size, kernel_initializer='he_normal',
                  name='dense2')(concatenate([lstm_2, lstm_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)

    # Handlers
    labels = Input(name='the_labels', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(callback_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # Build model
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    print model.summary()

    return model
