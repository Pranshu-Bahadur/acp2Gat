import tensorflow as tf
from pandas import read_csv, DataFrame, concat
from tensorflow.keras import Model, Sequential
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
from tensorflow.keras.layers import TextVectorization, Input, Embedding, Conv1D,\
 MultiHeadAttention, LayerNormalization, Add, Dense, Flatten, BatchNormalization,\
  DepthwiseConv1D, MaxPooling1D,\
   GlobalAveragePooling1D, Concatenate, GroupNormalization, LSTM, GlobalMaxPooling1D, Activation,\
    Dropout, Attention, Dot, Bidirectional, GRU
from tensorflow.keras.metrics import AUC
import numpy as np
import torch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, ReLU, LayerNormalization, RNN, SimpleRNNCell
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from torch.nn import RNN
import torch
from itertools import repeat

class Retention(Layer):
    def __init__(self, dim=128, seq_len = 50, gamma = 0.9865, **kwargs):
        super().__init__()
        _dense_kwargs = {
                "use_bias" : False,
                "dtype" : "float32"
                }
        self._qkv_dense = Dense(dim, **_dense_kwargs) #Using only one dense layer for all Q, K, V
        self.D = self._compute_decay(seq_len, gamma)
        self.seq_len = seq_len
        self.gamma = tf.cast(gamma, tf.float32)

    def call(self, x, training=False):
        Q, K, V = [f(z) for f, z in product(self._qkv_layers, x)]
        _, _, d = Q.shape
        x = Q@tf.transpose(K, perm=[0, 2, 1])
        x /= d**0.5 #Normalization Trick 1
        D = self.D
        D /= tf.reduce_sum(D, 1)**0.5 #Normalization Trick 2
        x = x*D
        _norm_3 = lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, 1)), 1))
        x = tf.vectorized_map(_norm_3, x) #Normalization Trick 3
        x = x@V
        return x
    
    def _compute_decay(self, seq_len, gamma = 0.96875):
        _indices = torch.arange(seq_len, dtype=torch.float)
        _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
        D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
        return tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])

class MultiScaleRetention(Layer):
    def __init__(self, dim, hdim=100, seq_len=50, retention_layer=Retention, **kwargs):
      super(MultiScaleRetention, self).__init__()
      gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))
      gamma = gamma.numpy().tolist()
      self.dim = dim
      self.hdim = hdim
      self.heads = [retention_layer(dim=hdim, gamma=gamma[head], seq_len=seq_len, **kwargs) for head in range(dim // hdim)]
      self.gn = GroupNormalization(scale=False)
      self.wg = Dense(dim, use_bias=True, activation = 'swish', **kwargs)
      self.wo = Dense(dim, use_bias=True, **kwargs)

    def call(self, q, k, v):
      W = self.wg(q)
      x = [q, k, v]
      q, k, v = list(map(lambda val: tf.split(q, self.dim//self.hdim, 2), x))
      x = [headi([qi, ki, vi]) for headi, qi, ki, vi in zip(self.heads, q, k, v)]
      x = tf.concat(x, -1)
      Y = self.gn(x)
      x = self.wo(W * Y)
      return x

