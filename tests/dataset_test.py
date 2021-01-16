import unittest

from deepseg.dataset import DatasetBuilder, TokenMapper, LabelMapper


class DatasetTest(unittest.TestCase):

    def test_dataset(self):
        token_mapper = TokenMapper(vocab_file='testdata/vocab_small.txt')
        label_mapper = LabelMapper()
        builder = DatasetBuilder(token_mapper, label_mapper)
        train_dataset = builder.build_train_dataset(
            input_files='testdata/train_small.txt',
            batch_size=4,
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
