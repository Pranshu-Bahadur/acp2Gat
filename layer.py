from tensforflow.keras.layers import Layer, Dense
from itertools import product

class GAT(Layer):
    def __init__(self, units, seq_len, ngrams=5, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.units = units
        self.seq_len = seq_len
        self.ngrams = ngrams
        self.mhas = list(map(MultiHeadAttention(2, self.ngrams//2), list(range(self.seq_len//self.ngrams))))

    def call(self, x, training=False):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.split(x, self.seq_len//self.ngrams, axis=2)
        alpha = list(map(lambda x, a: a(x, x), x, self.mhas))
        alpha = tf.concat(alpha, axis=2)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x
