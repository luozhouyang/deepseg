# deepseg

Tensorflow 2.x 实现的神经网络分词模型！一键训练&一键部署！

> tensorflow 1.x的实现请切换到`tf1`分支


## 开发环境

```bash
conda create -n deepseg python=3.6
conda activate deepseg 
pip install -r requirements.txt
```

## 数据集下载

* [SIGHAN](https://lzy-oss-files.oss-cn-hangzhou.aliyuncs.com/segmentation/sighan-icwb2-data.zip)

## 训练模型

可以使用`deepseg/run_deepseg.py`脚本来训练你的模型。需要提供以下参数：

* `--model`，模型，可选择 `bisltm-crf`, `bigru-crf`, `bert-crf`, `albert-crf`, `bert-bilstm-crf`, `albert-bilstm-crf`
    - 如果是`bert-based`或者`albert-based`模型，请提供预训练模型路径，使用`--pretrained_model_dir`参数制定。
* `--model_dir`，模型保存路径
* `--vocab_file`，词典文件路径，注意是**字符**级别的词典，参考`testdata/vocab_small.txt`
* `--train_input_files`，训练文件，分好词的文本文件，参考`testdata/train_small.txt`

对于`bilstm-crf`和`bigru-crf`模型，还需要指定以下参数：

* `--vocab_size`，词典大小
* `--embedding_size`，潜入层的维度

一个使用`bert-crf`模型的例子如下：

```bash
python -m deepseg.run_deepseg \
    --model=bert-crf \
    --model_dir=models/bert-crf-model \
    --pretrained_model_dir=/home/zhouyang.lzy/pretrain-models/chinese_roberta_wwm_ext_L-12_H-768_A-12 \
    --train_input_files=testdata/train_small.txt \
    --vocab_file=/home/zhouyang.lzy/pretrain-models/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt \
    --epochs=2 
```

什么？你觉得我的训练脚本写得太烂了，想自己写训练过程？

完全OK啊！

### 自己写训练脚本

```python
from deepseg.dataset import DatasetBuilder, LabelMapper, TokenMapper
from deepseg.models import BiGRUCRFModel, BiLSTMCRFModel
from deepseg.models import AlbertBiLSTMCRFModel, AlbertCRFModel
from deepseg.models import BertBiLSTMCRFModel, BertCRFModel

token_mapper = TokenMapper(vocab_file='testdata/vocab_small.txt')
label_mapper = LabelMapper()

builder = DatasetBuilder(token_mapper, label_mapper)
train_dataset = builder.build_train_dataset('testdata/train_small.txt', batch_size=20, buffer_size=100)
valid_dataset = None

model_dir = 'model/bilstm-crf'
tensorboard_logdir = os.path.join(model_dir, 'logs')
saved_model_dir = os.path.join(model_dir, 'export', '{epoch}')

# 更改成你自己想要的模型，或者干脆自己构建任何你想要的模型！
model = BiLSTMCRFModel(100, 128, 3)
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss' if valid_dataset is not None else 'loss'),
        tf.keras.callbacks.TensorBoard(tensorboard_logdir),
        tf.keras.callbacks.ModelCheckpoint(
            saved_model_dir,
            save_best_only=False,
            save_weights_only=False)
    ]
)
```

## 部署模型

上面训练过程中，每个epoch都会保存一个SavedModel格式的模型，可以直接使用`tensorflow-serving`部署。

* TODO：增加部署文档和客户端调用文档
