import tensorflow as tf


def build_train_dataset(hparams):
    features_file = hparams.features_file
    labels_file = hparams.labels_file

    features_dataset = tf.data.TextLineDataset(features_file)
    labels_dataset = tf.data.TextLineDataset(labels_file)

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

    iterator = dataset.make_initializable_iterator()

    return iterator


def build_validation_dataset(hparams):
    features_file = hparams.validation_features_file
    labels_file = hparams.validation_labels_file

    features_dataset = tf.data.TextLineDataset(features_file)
    labels_dataset = tf.data.TextLineDataset(labels_file)

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

    iterator = dataset.make_initializable_iterator()

    return iterator
