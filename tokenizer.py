from typing import List, Tuple, Dict
from tensorflow.keras.layers import TextVectorization

class Tokenizer(object):
    def __init__(self, max_length : int, vocabs : List[List[str]], **kwargs):
        self.tokenizers = dict(map(lambda vocab: (
            len(vocab[-1]),
            self._generate_tokenizer(max_length,
                                     vocab)),
                                   vocabs))
        self.max_length = max_length

    def _generate_tokenizer(self, max_length : int,
                            vocab : List[str],
                            **kwargs) -> Tuple[int, TextVectorization]:
        _tv_kwargs = {
                'split' : 'character',
                'ngrams' : len(vocab[-1]),
                'output_mode' : 'int',
                'output_sequence_length' : max_length,
                'vocabulary' : vocab
                }
        tokenizer = TextVectorization(**_tv_kwargs)
        return tokenizer
