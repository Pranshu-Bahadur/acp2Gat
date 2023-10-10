from typing import List
from pandas import read_csv
from tensorflow.keras.layers import TextVectorization
from functools import reduce
from itertools import repeat, product

class ACP2Dataset(object):
    def __init__(self, fpath : str, columns=['text', 'label']):
        _df = read_csv(fpath, sep='\t')[columns]
        self.sequences = _df[columns[0]]
        self.labels = _df[columns[1]]

class ACP2TrainDataset(ACP2Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_amino_acids(self) -> List[str]:
        max_length = max(self.sequences.str.len())
        _tv_kwargs = {
                'split' : 'character',
                'ngrams' : 1,
                'output_mode' : 'int',
                'output_sequence_length' : max_length,
                #'pad_to_max_tokens' : True,
                #'max_tokens' : max_length,
                }
        _tv_layer = TextVectorization(**_tv_kwargs)
        _tv_layer.adapt(self.sequences)
        return _tv_layer.get_vocabulary()[2:]

    def generate_vocab(self, ngrams : int = 1) -> List[List[str]]:
        _amino_acids = self.generate_amino_acids()
        _result = list(map(lambda ngram: product(*list(repeat(_amino_acids, ngram+1))), list(range(ngrams))))
        _result = list(map(lambda x: list(map(lambda y: ''.join(y), x)), _result))
        return [reduce(lambda x, y: x+y, _result[:i], []) for i in range(1, ngrams+1)] #@TODO: add option for non-reduced vocab
