from keras_core.layers import GroupNormalization
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, \
 LeakyReLU, LayerNormalization
from tensorflow.keras import Sequential
from itertools import product, repeat
import torch


class Retention(Layer):
    def __init__(self, dim=128, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
        super().__init__()
        _dense_kwargs = {
                "use_bias" : False,
                "dtype" : "float32"
                }
        self._qkv_layers = [*repeat(Dense(dim, **_dense_kwargs), 3)]
        self.D = self._compute_decay(seq_len, gamma)
        self.seq_len = seq_len
        self.gamma = tf.cast(gamma, tf.float32)

    def call(self, x, training=False):
        Q, K, V = [f(z) for f, z in zip(self._qkv_layers, x)]
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

class GAT(Layer):
    def __init__(self, units, seq_len, ngrams=5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seq_len = seq_len
        self.ngrams = ngrams

        """
        self.mhas = list(map(lambda x: Retention(dim=self.ngrams, 
        seq_len=self.units), 
        list(range(self.seq_len//self.ngrams))))
        list(map(lambda x: 
        """

        self.mhas = MultiHeadAttention(num_heads=self.seq_len//self.ngrams, 
        key_dim=self.ngrams,
        value_dim=self.ngrams,
        dropout=0.1)


        
 
        
    def call(self, x, training=False):
      shape = tf.shape(x)
      x = tf.transpose(x, perm=[0, 2, 1])
      """
      #x = tf.split(x, self.seq_len//self.ngrams, axis=2)
      nodes = list(map(lambda node: tf.concat(node, -1),
       list(product(x, repeat=3))))
      print(nodes)
      """
      """
      nodes = list(map(lambda subset: list(map(lambda edges: tf.concat(edges, 2),
       list(product(tf.split(subset, 1, 2), repeat=3)))), x))

      """
      alpha = self.mhas(*repeat(x, 3))
      """
      list(map(lambda a, h: a(h, h, h),
       self.mhas, nodes))
      """
      print(alpha)
      x = tf.concat(alpha, -1)
      print(x.shape)

      #x = tf.reshape(x, (x.shape[0], x.shape[1], -1))
      #x = self.norm(x)
      x = tf.transpose(x, perm=[0, 2, 1])
      #x = tf.reduce_mean(x, -1)
      return x
