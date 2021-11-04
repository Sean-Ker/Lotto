#%% https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
import random, os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.layers import (Dense, Lambda, Layer, Dropout, TimeDistributed, Add,
                                     Concatenate, Conv1D, GlobalMaxPool1D, Conv2D, Input, Embedding, Activation, Conv1D, LayerNormalization, Softmax)
from tensorflow.keras.models import Model
# from utils.attention import ScaledDotProductAttention
from tensorflow.keras.utils import plot_model
import numpy as np

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
        return result

class MultiHeadAttention(Layer):
    def __init__(self, d_model, n_head, dropout_rate, attention_dropout=None):
        super(MultiHeadAttention, self).__init__()
        assert not d_model % n_head
        self.depth = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        if not attention_dropout:
            attention_dropout = dropout_rate

        self.sdpa = ScaledDotProductAttention(dropout = attention_dropout, train_scale=True) ### CHANGE LATER!! Also, remember you're awesome(: And I really love you man. Truly do. You are an amzing woman with a great resume.
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)
        return

    def split_heads(self, x, batch_size):
        """Split the last 2 dimensions into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.depth)) # [batch_size, seq_len, heads, (d_model // heads)]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, return_attn=True):
        assert len(x) == 3
        q, k, v = x
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = self.sdpa([q, k, v], return_weights=return_attn)
        if return_attn:
            scaled_attention, attention_weights = scaled_attention

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])             # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)                                            # (batch_size, seq_len_q, d_model)
        output = self.dropout(output)
        # return output, attention_weights if return_attn else output # output[batch, T, d_model] | weights[batch, num_heads, T, T]
        if return_attn:
            return output, attention_weights
        return output


# import numpy as np
# i = np.random.randn(2, 40, 161).astype(np.float32)
# i = Input(i.shape[1:])
# mha = MultiHeadAttention(512, 8, .1)
# x = mha([i,i,i], return_attn=True)
# model = Model(i, x)
# model.summary()
# plot_model(model, show_shapes=True, expand_nested=True, )
# x[1]
# model = Model(inp, x)
#%%
class PointwiseFeedForward(Layer):
    def __init__(self, d_out, dff, dropout):
        super(PointwiseFeedForward, self).__init__()
        self.dense1 = Conv1D(dff, 1, activation='relu') # (batch_size, seq_len, dff)
        self.dense2 = Conv1D(d_out, 1)                  # (batch_size, seq_len, d_model)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def call(self, x):
        output = self.dense1(x)
        output = self.dense2(output)
        return self.dropout(output + x)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, multiplier = 1):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.multiplier = multiplier

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * self.multiplier

class PositionalEncoder(Layer):
    def __init__(self):
        super(PositionalEncoder, self).__init__()

    def build(self, inp_shape):
        # print(inp_shape)
        _, self.position, self.d_model = inp_shape

    def get_angles(self):
        pos = np.arange(self.position)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = pos / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return angle_rates

    def call(self, x):
        angle_rads = self.get_angles()
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...] + x
        return tf.cast(pos_encoding, dtype=tf.float32)
# TEST
# a = np.random.random((1,40,150)).astype(np.float32)
# pos_encoding = PositionalEncoding()(a)
# plt.pcolormesh(pos_encoding[-0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
# plt.pcolormesh((pos_encoding-a)[-0], cmap='RdBu')
# plt.colorbar()
#%

class EncoderLayer(Layer):
    def __init__(self, d_model, d_inner_hid, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.pos_ffn_layer  = PointwiseFeedForward(d_model, d_inner_hid, dropout)
        self.mha = MultiHeadAttention(d_model, n_head, dropout)
        self.norm_layer = LayerNormalization(epsilon=1e-6)

    def call(self, x, return_attn= True, force_dims = True):
        output = self.mha([x,x,x], return_attn = return_attn)
        if return_attn:
            output, slf_attn = output
        if force_dims and x.shape[-1] != self.d_model:
            # print('Warning: Reshaped inputs to match d_model')
            x = Dense(self.d_model, use_bias=False)(x)
        out1 = self.norm_layer(x + output) # (batch_size, input_seq_len, d_model) # seq_len = D!!

        out2 = self.pos_ffn_layer(out1)
        out2 = self.norm_layer(out1 + out2) # (batch_size, input_seq_len, d_model)
        if return_attn:
            return out2, slf_attn
        return out2

class Encoder(Layer):
    def __init__(self, d_model=512, d_inner_hid=2048, n_head=8, n_layer=4, dropout=0.1,  pos_encode = True):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.enc_layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(n_layer)]
        self.dropout = Dropout(dropout)
        self.pos_encode = pos_encode

    def call(self, x, return_attn=True):
        '''
        Inputs:
        [batch_size, T, D]'''
        if self.pos_encode:
            x = PositionalEncoder()(x)
        x = self.dropout(x)

        if return_attn:
            atts = []
            for i in range(self.n_layer):
                x, att = self.enc_layers[i](x, True)
                atts.append(att)
            return x, atts

        for i in range(self.n_layer):
            x = self.enc_layers[i](x, False)
        return x  # (batch_size, input_seq_len, d_model)

# a = np.random.random((2,40,512)).astype(np.float32)
# inp = Input(a.shape[1:])
# Encoder(256, 512,8,4,.1, pos_encode = True)(a, return_attn = False)
# x = EncoderLayer(256, 512,8,.1)(a, return_attn = False)
# x.shape, len(attns)
# x.shape
# model = Model(inp, attns)
# # model(a)

# model.summary()
# plot_model(model, to_file='model3.png', show_shapes = True, expand_nested = True, dpi=300)
# dot = tf.keras.utils.model_to_dot(model, show_shapes = True, expand_nested = True, subgraph=True)
# model.compile('adam','mse')
# model.predict(a)
# %%
# if __name__ == '__main__':
#     i = Input(())
#     x = SelfAttention((i)

if __name__ == "__main__":
    def test(word):
        chars = [char for char in word]
        return chars

    i, *args = test('asdf')
    i, args

    l = []
    # @tf.function
    class f(Layer):
        def __init__(self):
            super(f, self).__init__()
        def call(self, x):
            l = []
            for i in x:
                l.append(i + 1)    # Caution! Will only happen once when tracing
            return l

    f()([1, 2, 3])


# %%
