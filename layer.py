from keras_core.layers import GroupNormalization
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, \
 LeakyReLU, LayerNormalization
from tensorflow.keras import Sequential
from itertools import product, repeat
import torch


class GAT(Layer):
    def __init__(self, units, seq_len, ngrams=5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seq_len = seq_len
        self.ngrams = ngrams

        self.mhas = list(map(lambda x: MultiHeadAttention(num_heads=self.ngrams, 
        key_dim=2,
        value_dim=2, 
        dropout=0.2), 
        list(range(self.seq_len//self.ngrams))))
        self.activ = LeakyReLU()

    def call(self, x, training=False):
      shape = tf.shape(x)
      x = tf.transpose(x, perm=[0, 2, 1])
      x = tf.split(x, self.seq_len//self.ngrams, axis=2)
      nodes = list(map(lambda subset: list(map(lambda edges: 
      tf.concat(list(map(lambda a,b : tf.concat([a, b], -1), 
      tf.split(edges[0], edges[0].shape[-1], 2),
       tf.split(edges[1], edges[1].shape[-1], 2)
      )), -1),
       list(product(tf.split(subset, 1, 2), repeat=2)))), x))
      alpha = list(map(lambda a, h: self.activ(a(*h, *h)), self.mhas, nodes))
      x = tf.stack(alpha, -1)
      x = tf.reduce_mean(x, 2)
      x = tf.transpose(x, perm=[0, 2, 1])
      x = tf.reduce_mean(x, -1)
      return x
