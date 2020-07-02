import tensorflow as tf
from naivenlp.tokenizers.abstract_tokenizer import VocabBasedTokenizer

LABEL_MAP = {
    'S': 1,
    'B': 2,
    'I': 3,
    'E': 4,
}


class DatasetBuilder(object):

    def __init__(self, tokenizer: VocabBasedTokenizer, max_len=100):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _labeling_fn(self, x):

        def _labeling(example):
            text = example.numpy().decode('utf-8')
            tokens, tags = [], []
            for token in text.split():
                if len(token) == 1:
                    tokens.append(token)
                    tags.append('S')
                    continue
                for i in range(len(token)):
                    if i == 0:
                        tokens.append(token[i])
                        tags.append('B')
                        continue
                    if i == len(token) - 1:
                        tokens.append(token[i])
                        tags.append('E')
                        continue
                    tokens.append(token[i])
                    tags.append('I')
            input_ids = [self.tokenizer.vocab.get(x, self.tokenizer.unk_id) for x in tokens]
            label_ids = [LABEL_MAP.get(x, 0) for x in tags]
            return input_ids, len(input_ids), label_ids

        sequence, sequence_length, labels = tf.py_function(_labeling, [x], [tf.int64, tf.int64, tf.int64])
        return sequence, sequence_length, labels

    def build_train_dataset(
            self,
            train_files,
            batch_size=32,
            skip_count=0,
            repeat=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            padded_to_max_len=False,
            **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(train_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TextLineDataset(x).skip(skip_count),
            cycle_length=len(train_files),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle)
        dataset = dataset.map(lambda x: self._labeling_fn(x), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda x, y, z: tf.size(x) < self.max_len)
        padded_len = self.max_len if padded_to_max_len else None
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([padded_len], [], [padded_len]),
            padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))
        ).prefetch(batch_size)
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
        return dataset

    def build_valid_dataset(
            self,
            valid_files,
            batch_size=32,
            skip_count=0,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            padded_to_max_len=False,
            **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(valid_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TextLineDataset(x).skip(skip_count),
            cycle_length=len(valid_files),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle)
        dataset = dataset.map(lambda x: self._labeling_fn(x), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda x, y, z: tf.size(x) < self.max_len)
        padded_len = self.max_len if padded_to_max_len else None
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([padded_len], [], [padded_len]),
            padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))
        ).prefetch(batch_size)
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
        return dataset

    def _predict_map_fn(self, x):

        def _map(example):
            text = example.numpy().decode('utf-8')
            tokens = text.split()
            input_ids = [self.tokenizer.vocab.get(x, self.tokenizer.unk_id) for x in tokens]
            return input_ids, len(input_ids)

        seq, seq_len = tf.py_function(_map, [x], [tf.int64, tf.int64])
        return seq, seq_len

    def build_predict_dataset(
            self,
            predict_files,
            batch_size=32,
            skipt_count=0,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            padded_to_max_len=False,
            **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(predict_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TextLineDataset(x).skip(skipt_count),
            num_parallel_calls=num_parallel_calls
        )
        dataset = dataset.map(lambda x: self._predict_map_fn(x), num_parallel_calls=num_parallel_calls)
        padded_len = self.max_len if padded_to_max_len else None
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([padded_len], []),
            padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))
        ).prefetch(batch_size)
        dataset = dataset.map(lambda x, y: ((x, y), None))
        return dataset
