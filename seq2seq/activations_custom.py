import keras
from keras.layers import Layer
import keras.backend as K
from keras.activations import softmax

def smoothmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.sigmoid(x)
    elif ndim > 2:
        e = K.sigmoid(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
'Received input: %s' % x)

def sharpmax(x, B, axis=-1):
    """Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim > 1 :
        e = K.exp(x * B)
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
'Received input: %s' % x)

#Keras 2.1.0 does not support Softmax function layer, redefinition of Softmax layer as in Keras 2.3
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
        return softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

#Keras 2.1.0 does not support Softmax function layer, redefinition of Softmax layer as in Keras 2.3
class Smoothmax(Layer):
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
        super(Smoothmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return smoothmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Smoothmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

#Keras 2.1.0 does not support Softmax function layer, redefinition of Softmax layer as in Keras 2.3
class Sharpmax(Layer):
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

    def __init__(self, axis=-1, sharpenning = 2.0, **kwargs):
        super(Sharpmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.sharp = sharpenning

    def call(self, inputs):
        return sharpmax(inputs, self.sharp, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Sharpmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

