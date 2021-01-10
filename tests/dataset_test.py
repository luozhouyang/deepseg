import unittest

from deepseg.dataset import DatasetBuilder


class DatasetTest(unittest.TestCase):

    def test_dataset(self):
        builder = DatasetBuilder(vocab_file='testdata/vocab.txt', unk_token='[UNK]', pad_token='[PAD]')
        train_dataset = builder.build_train_dataset(
            input_files='testdata/train.txt',
            batch_size=2,
            buffer_size=100,
            repeat=100, )
        print(next(iter(train_dataset)))


if __name__ == "__main__":
    unittest.main()
