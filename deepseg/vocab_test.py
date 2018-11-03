import tensorflow as tf

from . import vocab


class VocabTest(tf.test.TestCase):

    def testGenerateFeatureVocab(self):
        feat_file = "testdata/feature.txt"
        feat_vocab_file = "testdata/vocab.feature.txt"
        vocab.generate_feature_vocab(feat_file, feat_vocab_file, vocab_limit=100)

    def testGenerateLabelVocab(self):
        feat_file = "testdata/label.txt"
        feat_vocab_file = "testdata/vocab.label.txt"
        vocab.generate_label_vocab(feat_vocab_file)

    def testCreateVocabTable(self):
        feat_vocab_file = "testdata/vocab.feature.txt"
        word2idx, idx2word = vocab.create_vocab_table(feat_vocab_file)
        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            idx = word2idx.lookup(tf.constant(vocab.UNK, dtype=tf.string))
            self.assertEqual(0, sess.run(idx))
            word = idx2word.lookup(tf.constant(vocab.UNK_ID, dtype=tf.int64))
            self.assertEqual(vocab.UNK, sess.run(word).decode("utf8"))


if __name__ == "__main__":
    tf.test.main()
