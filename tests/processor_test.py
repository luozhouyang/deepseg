import unittest

from deepseg.processor import SIGHANDataProcessor


class ProcessorTest(unittest.TestCase):

    def test_sighan_processor(self):
        p = SIGHANDataProcessor('testdata/icwb2-data', pad_token='[PAD]', unk_token='[UNK]')
        p.process(
            train_output_file='testdata/sighan/train.txt',
            test_features_output_file='testdata/sighan/test_features.txt',
            test_labels_output_file='testdata/sighan/test_labels.txt',
            words_output_file='testdata/sighan/words.txt',
            vocab_output_file='testdata/sighan/vocab.txt')


if __name__ == "__main__":
    unittest.main()
