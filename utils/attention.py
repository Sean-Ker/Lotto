# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-672-lb/notebook#Training
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, Input, Layer, Conv2D, Softmax)
# from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model, Sequential

#%%
class ScaledDotProductAttention(Layer):
    """Dot-product attention layer, a.k.a. Luong-style attention.
    Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
        shape `[batch_size, Tv, dim]` and `key` tensor of shape
        `[batch_size, Tv, dim]`.
    The calculation follows the steps:
        1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
            product: `scores = tf.matmul(query, key, transpose_b=True)`.
        2. Use scores to calculate a distribution with shape
            `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
        3. Use `distribution` to create a linear combination of `value` with
            shape `[batch_size, Tq, dim]`:
            `return tf.matmul(distribution, value)`.
    Output shape:
        Attention outputs of shape `[batch_size, Tq, dim]`
        Weights of shape `[batch_size, Tv, dim]`"""
    def __init__(self, dropout=0.0, train_scale=True, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.train_scale = train_scale
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        """Creates scale variable if use_scale==True."""
        if self.train_scale:
            constant = tf.keras.initializers.Constant(input_shape[-1][-1]**(-0.5))
            self.scale = self.add_weight(trainable=True, initializer=constant)
        else:
            self.scale = 1/tf.sqrt(tf.constant(input_shape[-1][-1], dtype='float32'))
        super(ScaledDotProductAttention, self).build(input_shape)

    def get_config(self):
        config = {'train_scale': self.train_scale, 'dropout': self.dropout}
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, return_weights=True):
        '''
        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        '''
        assert len(inputs) == 3
        q, k, v = inputs[0], inputs[1], inputs[2]

        scores = tf.matmul(q, k, transpose_b=True)      # [batch_size, ..., seq_len_q, seq_len_k]
        scores *= self.scale

        weights = Softmax()(scores)
        weights = self.dropout(weights)                 # [batch_size, ..., Tv, dim]
        result = tf.matmul(weights, v)                  # [batch_size, ..., Tq, dim]

        if return_weights:
            return result, weights
        else:
            return result


# TESTING
q = tf.random.normal((2, 5, 10))
k = tf.random.normal((2, 7, 10))
v = tf.random.normal((2, 7, 10))
inp = [q,k,v]
x = ScaledDotProductAttention(dropout = .1)(inp)
# y = ScaledDotProductAttention_old()(inp)
import numpy as np
# np.isclose(x[0],y[0])
x[0].shape

# q = Input((5,10))
# k = Input((7, 10))
# v = Input((7, 10))
# inp = [q,k,v]
# x = ScaledDotProductAttention_old()(inp)
# model = Model(inp,x)
# model.summary()
# plot_model(model, show_shapes = True)


#%%
class TemporalPatternAttentionMechanism(Layer):
    """
    Input:
    query: [batch_size, attn_size * 2] (c and h)
    attn_states: [batch_size, attn_length, attn_size] (h)

    Output:
    new_attns: [batch_size, attn_size]
    new_attn_states: [batch_size, attn_length - 1, attn_size]
    """
    def __init__(self, filter_num=32, filter_size = 1):
        super(TemporalPatternAttentionMechanism, self).__init__()
        self.filter_num = filter_num
        self.filter_size = filter_size

    def bulid(self, query_shape, attn_states_shape):
        print('i was here')
        _, self.attn_length, self.attn_size = attn_states_shape

        self.feature_dim = self.attn_size - self.filter_size + 1

        # w: [batch_size, 1, filter_num]
        self.w = Dense(self.filter_num, use_bias=False)
        self.w = tf.reshape(self.w, [-1, 1, self.filter_num])

        # conv_vecs: [batch_size, feature_dim, filter_num]
        self.conv_vecs = Conv2D (filters=self.filter_num, kernel_size=[self.attn_length, self.filter_size], padding="valid", activation=None)
        self.conv_vecs = tf.reshape(self.conv_vecs, [-1, self.feature_dim, self.filter_num])

    def call(self, query, attn_states):
        _, self.attn_length, self.attn_size = attn_states.shape

        assert query.shape[1]//2 == self.attn_size

        self.feature_dim = self.attn_size - self.filter_size + 1

        # w: [batch_size, 1, filter_num]
        w = Dense(self.filter_num, use_bias=False)(query)
        print(w)
        w = tf.reshape(w, (1, self.filter_num))

        reshape_attn_vecs = tf.reshape(attn_states,[-1, self.attn_length, self.attn_size, 1])

        # conv_vecs: [batch_size, feature_dim, filter_num]
        conv_vecs = Conv2D(filters=self.filter_num, kernel_size=[self.attn_length, self.filter_size], padding="valid", activation=None)(reshape_attn_vecs)
        conv_vecs = tf.reshape(conv_vecs, (self.feature_dim, self.filter_num))

        # s: [batch_size, feature_dim]
        s = tf.multiply(conv_vecs, w)
        s = tf.math.reduce_sum(s, 1)

        # a: [batch_size, feature_dim]
        a = tf.sigmoid(s)

        # d: [batch_size, filter_num]
        d = tf.multiply(tf.reshape(a, [-1, self.feature_dim, 1]), conv_vecs)
        d = tf.math.reduce_sum(d, [1])
        new_conv_vec = tf.reshape(d, [-1, self.filter_num])
        new_attns = Dense(self.attn_size)(tf.concat([query, new_conv_vec], axis=1))
        new_attn_states = tf.slice(attn_states, [0, 1, 0], [-1, -1, -1])
        return new_attns, new_attn_states



# %% Work under progress.

# class TemporalPatternAttentionModel(Model):
#     def __init__(self, num_layers, attn_length, attn_size=None, attn_vec_size=None):
#         ''' Args:
#                 cell: an RNNCell, an attention is added to it.
#                 attn_length: integer, the size of an attention window.
#                 attn_size: integer, the size of an attention vector. Equal to
#                     cell.output_size by default.
#                 attn_vec_size: integer, the number of convolutional features
#                     calculated on attention state and a size of the hidden layer
#                     built from base cell state. Equal attn_size to by default.
#                 input_size: integer, the size of a hidden linear layer, built from
#                     inputs and attention. Derived from the input tensor by default.'''
#     super(TemporalPatternAttentionModel, self).__init__()
#     if attn_length <= 0:
#             raise ValueError("attn_length should be greater than zero, got %s" % str(attn_length))
#     if attn_size is None:
#         self.attn_size = cell.output_size
#     if attn_vec_size is None:
#         attn_vec_size = attn_size

#     self.num_layer = num_layers


    # def call(self, inputs):


#%% Bahdanau (Concat) Additive Attention 2015
class AttentionBahdanau(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionBahdanau, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == self.step_dim

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        self.W = K.reshape(self.W, (self.features_dim, 1))
        x = K.reshape(x, (-1, self.features_dim))

        eij = K.dot(x, self.W)
        eij = K.reshape(eij, (-1, self.step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a_sum = K.sum(a, axis=1, keepdims=True) + K.epsilon()
        a_sum = K.cast(a_sum, K.floatx())
        a = a / a_sum

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_config(self):
        return super(AttentionBahdanau, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



# check to see if it compiles
if __name__ == '__main__':
    i = Input(shape=(100, 104), dtype='float32')
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
    dec = AttentionDecoder(units=100, output_dim=4)(enc)
    dec2 = AttentionBahdanau(100)(enc)

    model = Model(inputs=i, outputs=dec)
    model2 = Model(i, dec2)
    print(model.summary())
    print(model2.summary())


    attn_size = 101
    attn_length = 10

    inp = Input(shape=(attn_length,attn_size))

    attn_states, h, c = LSTM(attn_size, return_sequences=True, return_state = True)(inp)
    query = tf.concat([c,h], axis = 1)

    x = TemporalPatternAttentionMechanism(filter_num = 32, filter_size = 1)(query, attn_states)

    model = Model(inp, x)
    model.summary()
