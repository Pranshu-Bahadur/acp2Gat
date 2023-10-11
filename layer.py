from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, \
 LeakyReLU, LayerNormalization
from tensorflow.keras import Sequential
from itertools import product

class GAT(Layer):
    def __init__(self, units, seq_len, ngrams=5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seq_len = seq_len
        self.ngrams = ngrams

        self.mhas = list(map(lambda x: MultiHeadAttention(num_heads=self.ngrams, 
        key_dim=2,
        value_dim=2), 
        list(range(self.seq_len//self.ngrams))))
        self.norm = Sequential([
          LeakyReLU(),
          ])

    def call(self, x, training=False):
      shape = tf.shape(x)
      x = tf.transpose(x, perm=[0, 2, 1])
      x = tf.split(x, self.seq_len//self.ngrams, axis=2)
      nodes = list(map(lambda subset: list(map(lambda edges: tf.concat(edges, 2),
       list(product(tf.split(subset, 1, 2), repeat=2)))), x))
      alpha = list(map(lambda a, h: self.norm(a(*h, *h)), self.mhas, nodes))
      alpha = tf.stack(alpha, -1)
      x = tf.reduce_mean(alpha, -1)
      x = tf.transpose(x, perm=[0, 2, 1])
      return x
