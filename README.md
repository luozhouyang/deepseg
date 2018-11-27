# deepseg

Chinese word segmentation in tensorflow.


## Architecture

The architecture of this model is simple. There are three components of the model:

* Embedding: words embedding layer
* BiLSTM: a bidirectional LSTM layer
* CRF: a conditional random field layer

Segmentation is some kind of tagging. We can tag each token of the input sequence with Only a few tags:

* B: begin of a token
* M: middle of a token
* E: end of a token
* S: single character as a token
* O: Out of tags

We train the model to tag every input sequence, and then wo process the tagged result, so we get the final segmentation.

## Training

Assuming that we have a hparams file in `deepseg/example_params.json`:

```bash
python -m deepseg.runner \
    --params_file=deepseg/example_params.json \
    --mode=train
```

## Eval

```bash
python -m deepseg.runner \
    --params_file=deepseg/example_params.json \
    --mode=eval
```

## Predict

```bash
python -m deepseg.runner \
    --params_file=deepseg/example_params.json \
    --mode=predict
```

## Train and eval

```bash
python -m deepseg.runner \
    --params_file=deepseg/example_params.json \
    --mode=train_and_eval
```

## Export

You may want to export the model to saved model format and serve it on tf serving, you can just run:

```bash
python -m deepseg.runner \
    --params_file=deepseg/example_params.json \
    --mode=export
```