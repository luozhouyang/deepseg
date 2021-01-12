import unittest

from deepseg.dataset import DatasetBuilder


class DatasetTest(unittest.TestCase):

    def test_dataset(self):
        builder = DatasetBuilder(vocab_file='testdata/sighan/vocab.txt', unk_token='[UNK]', pad_token='[PAD]')
        train_dataset = builder.build_train_dataset(
            input_files='testdata/train_small.txt',
            batch_size=2,
            buffer_size=100,
            repeat=1, )
        print(next(iter(train_dataset)))

        test_dataset = builder.build_predict_dataset(
            input_files='testdata/test_small.txt',
            batch_size=2
        )
        print(next(iter(test_dataset)))


if __name__ == "__main__":
    unittest.main()
