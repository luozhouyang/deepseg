import tensorflow as tf

from .dataset_builder import DatasetBuilder

from naivenlp.tokenizers.abstract_tokenizer import VocabBasedTokenizer


class DatasetBuilderTest(tf.test.TestCase):

    def testDatasetBuilder(self):
        tokenizer = VocabBasedTokenizer(vocab_file='testdata/vocab.txt')
        builder = DatasetBuilder(tokenizer=tokenizer, max_len=100)

        train_dataset = builder.build_train_dataset(
            train_files=['testdata/train.txt'],
            batch_size=2,
            repeat=10,
            buffer_size=100,
            padded_to_max_len=True,
        )

        v = next(iter(train_dataset))
        self.assertEqual([2, 100], v[0][0].shape)
        self.assertEqual([2], v[0][1].shape)
        self.assertEqual([2, 100], v[1].shape)

        predict_dataset = builder.build_predict_dataset(
            predict_files=['testdata/train.txt'],
            batch_size=1,
            padded_to_max_len=True,
        )
        v = next(iter(predict_dataset))
        self.assertEqual([1, 100], v[0][0].shape)
        self.assertEqual([1], v[0][1].shape)


if __name__ == "__main__":
    tf.test.main()
