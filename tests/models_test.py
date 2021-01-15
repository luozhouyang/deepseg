import unittest

import numpy as np
from deepseg.dataset import DatasetBuilder, LabelMapper, TokenMapper
from deepseg.models import BiGRUCRFModel, BiLSTMCRFModel

token_mapper = TokenMapper(vocab_file='testdata/vocab_small.txt')
label_mapper = LabelMapper()


class ModelsTest(unittest.TestCase):

    def _build_inputs(self):
        builder = DatasetBuilder(token_mapper, label_mapper)
        train_dataset = builder.build_train_dataset('testdata/train_small.txt', batch_size=20, buffer_size=100)
        pred_dataset = builder.build_predict_dataset('testdata/test_small.txt', batch_size=10)
        return train_dataset, pred_dataset

    def test_bilstm_crf_model(self):
        model = BiLSTMCRFModel(100, 128, 3)

        train_dataset, pred_dataset = self._build_inputs()
        print(next(iter(train_dataset)))
        model.fit(train_dataset, epochs=2)

        for p in model.predict(pred_dataset, steps=1):
            e = np.argmax(p, axis=-1).tolist()
            print(label_mapper.decode(e))

    def test_bigru_crf_model(self):
        model = BiGRUCRFModel(100, 128, 3)
        train_dataset, pred_dataset = self._build_inputs()
        print(next(iter(train_dataset)))
        model.fit(train_dataset, epochs=2)

        for p in model.predict(pred_dataset, steps=1):
            e = np.argmax(p, axis=-1).tolist()
            print(label_mapper.decode(e))


if __name__ == "__main__":
    unittest.main()
