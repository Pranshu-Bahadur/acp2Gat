from tokenizer import Tokenizer
from dataset import ACP2Dataset, ACP2TrainDataset
import unittest
from unittest import TestCase

TEST_PATH = '~/Downloads/acp2 main test - Sheet1(1).tsv'
TRAIN_PATH = '~/Downloads/acp2_main_train - Sheet1(1).tsv'

class TestACP2Dataset(TestCase):
    
    def test_init(self):
        test_dataset = ACP2Dataset(TEST_PATH)
        self.assertEqual(test_dataset.sequences.size, test_dataset.labels.size)
        train_dataset = ACP2TrainDataset(TRAIN_PATH)
        self.assertEqual(max(test_dataset.sequences.str.len()), max(train_dataset.sequences.str.len()))

    def test_generate_vocab(self):
        dataset = ACP2TrainDataset(TRAIN_PATH)
        vocab = dataset.generate_vocab(3)
        self.assertEqual(len(vocab), 3)
        self.assertEqual(len(vocab[0]), 20)
        vocab = dataset.generate_vocab(1)
        self.assertEqual(len(vocab), 1)
        self.assertEqual(len(vocab[0]), 20)

class TestTokenizer(TestCase):
    def test_init(self):
        dataset = ACP2TrainDataset(TRAIN_PATH)
        vocab = dataset.generate_vocab(3)
        tokenizer = Tokenizer(25, vocab)
        self.assertEqual(len(tokenizer.tokenizers), 3)

if __name__ == '__main__':
    unittest.main()
