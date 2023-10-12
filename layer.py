from keras_core.layers import GroupNormalization
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, \
 LeakyReLU, LayerNormalization
from tensorflow.keras import Sequential
from itertools import product, repeat, combinations
import torch
from keras_nlp.layers import PositionEmbedding



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
        self.graph_weights = PositionEmbedding(sequence_length=self.seq_len//self.ngrams)
        _num_heads = 4
        self.graph_attention = MultiHeadAttention(key_dim=self.units//_num_heads,
         num_heads=_num_heads, 
         dropout=0.2)

    def call(self, x, training=False):
      shape = tf.shape(x)
      x = tf.transpose(x, perm=[0, 2, 1])
      chunks = tf.split(x, self.seq_len//self.ngrams, axis=2)
      subsets = list(map(lambda subset: combinations(tf.split(subset, self.ngrams, -1), r=2), chunks))
      nodes = list(map(lambda edges: list(map(lambda edge: tf.concat(edge, -1), list(edges))), subsets))
      alpha = list(map(lambda a, h: self.activ(tf.concat(h, -1) + a(tf.concat(h, -1), tf.concat(h, -1))), self.mhas, nodes))
      x = tf.stack(alpha, -2)
      print(x.shape)
      x = tf.reduce_mean(x, -1)
      x = tf.transpose(x, perm=[0, 2, 1])
      #x = tf.reduce_mean(x, -1)
      x = x + self.graph_weights(x)
      _attention = self.graph_attention(*repeat(x, 2))
      x = x + _attention
      print(x.shape)
      return x
