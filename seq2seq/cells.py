import recurrentshop
from recurrentshop.cells import *
from keras.models import Model
from keras.layers import Layer, Input, Dense, TimeDistributed, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras import backend as K
from keras.backend import print_tensor
import keras.activations

#Keras 2.1.0 does not support Softmax function, redefinition of Softmax layer as in Keras 2.3
class Softmax(Layer):
    """Softmax activation function.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        axis: Integer, axis along which the softmax normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return activations.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class LSTMDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        super(LSTMDecoderCell, self).__init__(**kwargs)
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim

    def build_model(self, input_shape):
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   use_bias=False)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,)

        z = add([W1(x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)
        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)
        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim}
        base_config = super(ExtendedRNNCell, self).get_config()
        config.update(base_config)
        return config

class AttentionDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        self.input_ndim = 3
        super(AttentionDecoderCell, self).__init__(**kwargs)
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim


    def build_model(self, input_shape):
        
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        
        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W3 = Dense(1,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   activation='tanh')
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer)

        C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
        _xC = concatenate([x, C])
        _xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)

        alpha = W3(_xC)
        alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
        alpha = Activation('softmax', name='alpha')(alpha)
        _x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])

        z = add([W1(_x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)

        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation, name='cellout')(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim}
        base_config = super(ExtendedRNNCell, self).get_config()
        config.update(base_config)
        return config

class AttentionDecoderCellA(AttentionDecoderCell):

    def __init__(self, hidden_dim=None, **kwargs):
        super(AttentionDecoderCellA, self).__init__(hidden_dim=hidden_dim, **kwargs)


    def build_model(self, input_shape):
        
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        
        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W3 = Dense(1,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        U  = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer)

        C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
        _xC = concatenate([x, C])
        _xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)

        alpha = W3(_xC)
        alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
        alpha = Activation('softmax', name='alpha')(alpha)
        _x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])

        z = add([W1(_x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)

        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation, name='cellout')(W2(h))

        return Model([x, h_tm1, c_tm1], [alpha, h, c])

class AltAttentionDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        self.input_ndim = 3
        super(AltAttentionDecoderCell, self).__init__(**kwargs)
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim


    def build_model(self, input_shape):

        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        X = Input(batch_shape=input_shape, name='input')
        #readout = Input(batch_shape=(input_shape[0], output_dim), name='readout')
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim), name = 'pv_output')
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim), name = 'pv_state')
        alpha_tm1 = Input(batch_shape=(input_shape[0],input_length,1), name = 'pv_alpha')
        a_tm1 = Reshape((input_length,))(alpha_tm1)
       
        W = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,name="dW")
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,name="dU")
        V = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,name="dV")
        dX = Dense(1,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseX")
        dE = Dense(input_length,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseE")
        dT = Dense(1,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseT")
        dA = Dense(input_length,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseA")

        _x = dX(X)
        _E = dE(c_tm1)
        _E = Reshape(target_shape=(input_length,))(_E)
        _A = dA(a_tm1)
        en = add([_x,_E,_A])
        en = Activation('tanh')(en)
        energy =dT(en)
        alpha = Softmax(axis=-2, name='alpha')(energy)
        alphaD = Identity(name='alphaD')(alpha)

        _X = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alphaD, X])
        _X = Reshape(target_shape=(input_dim,))(_X)
        y1 = V(_X)
        y2 = U(h_tm1)

        z = add([y1,y2])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation,name='i')(z0)
        f = Activation(self.recurrent_activation,name='f')(z1)

        c = add([multiply([f, c_tm1],name='f_gate'), multiply([i, Activation(self.activation)(z2)],name='i_gate')],name='c')

        o = Activation(self.recurrent_activation,name='o')(z3)
        h = multiply([o, Activation(self.activation)(c)],name='h')
        y = Activation(self.activation, name='cellout')(W(h))

        model = Model([X, h_tm1, c_tm1, alpha_tm1], [y, h, c, alpha])
        return model

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim}
        base_config = super(AltAttentionDecoderCell, self).get_config()
        config.update(base_config)
        return config
           
    @property
    def num_states(self):
        return 4

class AltAttentionDecoderCellD(AltAttentionDecoderCell):

    def __init__(self, **kwargs):
        super(AltAttentionDecoderCellD, self).__init__(**kwargs)

    def build_model(self, input_shape):
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        X = Input(batch_shape=input_shape, name='input')
        #readout = Input(batch_shape=(input_shape[0], output_dim), name='readout')
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim), name = 'pv_output')
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim), name = 'pv_state')
        alpha_tm1 = Input(batch_shape=(input_shape[0],input_length,1), name = 'pv_alpha')
        a_tm1 = Reshape((input_length,))(alpha_tm1)
       
        W = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,name="dW")
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,name="dU")
        V = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,name="dV")
        dX = Dense(1,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseX")
        dE = Dense(input_length,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseE")
        dT = Dense(1,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseT")
        dA = Dense(input_length,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, name="DenseA")
        _x = dX(X)
        _E = dE(c_tm1)
        _E = Reshape(target_shape=(input_length,))(_E)
        _A = dA(a_tm1)
        en = add([_x,_E,_A])
        en = Activation('tanh')(en)
        energy =dT(en)
        alpha = Softmax(axis=-2, name='alpha')(energy)
        alphaD = Identity(name='alphaD')(alpha)

        _X = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alphaD, X])
        _X = Reshape(target_shape=(input_dim,))(_X)
        y1 = V(_X)
        y2 = U(h_tm1)

        z = add([y1,y2])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation,name='i')(z0)
        f = Activation(self.recurrent_activation,name='f')(z1)

        c = add([multiply([f, c_tm1],name='f_gate'), multiply([i, Activation(self.activation)(z2)],name='i_gate')],name='c')

        o = Activation(self.recurrent_activation,name='o')(z3)
        h = multiply([o, Activation(self.activation)(c)],name='h')
        y = Activation(self.activation, name='cellout')(W(h))

        model = Model([X, h_tm1, c_tm1, alpha_tm1], [alphaD, h, c, alpha])
        return model

    def get_config(self):
        config = {'hidden_dim': self.hidden_dim}
        base_config = super(AltAttentionDecoderCellD, self).get_config()
        config.update(base_config)
        return config
           
    @property
    def num_states(self):
        return 4

def reduce_sum_mult(x,w):
    n = K.get_variable_shape(x)
    for i in range(n[0]):
        for j in range(n[1]):
            x[i][j] *= w[j]
    return reduce('sum',[x],axis=-1)

class WeightedMultiply(Layer):
    def __init__(self, **kwargs):
        super(WeightedMultiply, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = ()
        for i in range(1,len(input_shape)-1):
            input_dim = input_dim + (input_shape[i],)
        initial_weight_value = np.random.random(input_dim)
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]


    def call(self, x, mask=None):
        return Lambda(lambda x,w: reduce_sum_mult(x,w), arguments={'w':self.W})(x)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2]

    def get_output_shape_for(self, input_shape):
        return input_shape[:-2]

#Just in case:
        #a_tm1 = Lambda(lambda x:print_tensor(x,message='Atm1'))(alpha_tm1)
        # n = input_shape[1]/2
        # x, c = Lambda(lambda x: [x[:,:n],x[:,n:]], mask = [None,None])(X)
        # def slice(a,b):
        #     return a[:,0,:]
        # L = Lambda(lambda a,b: slice(a,b), output_shape=(1,input_shape[2]), arguments={'b':X}, mask=None, name='lambda_slice')
        
        # E_tm1 = L(c)

# def AttentionMecanism(Layer):
#     """
#     # Input shape
#         nD tensor with shape: `(batch_size, ..., input_dim)`.
#         The most common situation would be
#         a 2D input with shape `(batch_size, input_dim)`.
#     # Output shape
#         nD tensor with shape: `(batch_size, ..., units)`.
#         For instance, for a 2D input with shape `(batch_size, input_dim)`,
#         the output would have shape `(batch_size, units)`.
#     """

#     @interfaces.legacy_dense_support
#     def __init__(self, units,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(Dense, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_spec = InputSpec(min_ndim=2)
#         self.supports_masking = True

#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]

#         self.kernel = self.add_weight(shape=(input_dim, self.units),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.units,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
#         self.built = True

#     def call(self, inputs):
#         output = K.dot(inputs, self.kernel)
#         if self.use_bias:
#             output = K.bias_add(output, self.bias)
#         if self.activation is not None:
#             output = self.activation(output)
#         return output

#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1]
#         output_shape = list(input_shape)
#         output_shape[-1] = self.units
#         return tuple(output_shape)

#     def get_config(self):
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer': regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Dense, self).get_config()
# return dict(list(base_config.items()) + list(config.items()))