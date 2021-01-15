import unittest

from deepseg.dataset import DatasetBuilder, LabelMapper, TokenMapper
from deepseg.models import (AlbertBiLSTMCRFModel, AlbertCRFModel,
                            BertBiLSTMCRFModel, BertCRFModel)

BERT_MODEL_DIR = '/home/zhouyang.lzy/pretrain-models/chinese_roberta_wwm_ext_L-12_H-768_A-12'
ALBERT_MODEL_DIR = '/home/zhouyang.lzy/pretrain-models/albert_base_zh'

token_mapper = TokenMapper(vocab_file='testdata/vocab_small.txt')
label_mapper = LabelMapper()


class PretrainedModelsTest(unittest.TestCase):

    def _build_inputs(self):
        builder = DatasetBuilder(token_mapper, label_mapper)
        train_dataset = builder.build_train_dataset('testdata/train_small.txt', batch_size=25, buffer_size=100)
        pred_dataset = builder.build_predict_dataset('testdata/test_small.txt', batch_size=10)
        return train_dataset, pred_dataset

    def test_bert_crf_model(self):
        model = BertCRFModel(BERT_MODEL_DIR)
        train_dataset, pred_dataset = self._build_inputs()
        model.fit(train_dataset, validation_data=train_dataset, epochs=2)

    def test_bert_bilstm_crf_model(self):
        model = BertBiLSTMCRFModel(BERT_MODEL_DIR)
        train_dataset, pred_dataset = self._build_inputs()
        model.fit(train_dataset, validation_data=train_dataset, epochs=2)

    def test_albert_crf_model(self):
        model = AlbertCRFModel(ALBERT_MODEL_DIR)
        train_dataset, pred_dataset = self._build_inputs()
        model.fit(train_dataset, validation_data=train_dataset, epochs=2)

    def test_albert_bilstm_crf_model(self):
        model = AlbertBiLSTMCRFModel(ALBERT_MODEL_DIR)
        train_dataset, pred_dataset = self._build_inputs()
        model.fit(train_dataset, validation_data=train_dataset, epochs=2)


if __name__ == "__main__":
    unittest.main()
