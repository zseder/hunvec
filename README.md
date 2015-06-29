## Why use hunvec?

hunvec is being developed to use neural networks in various nlp tasks.
Our intention is to support researchers to give them a tool with which one can experiment with different settings to create neural networks.
It is built upon pylearn2/theano, so recent advances in deep learning, that are supported in pylearn2, will hopefully work out of the box.
Now it supports basic sequential tagging network based on Natural Language Processing from Scratch paper, Collobert et al. 2011.
We designed hunvec in a way to be easily reconfigurable (adding, removing layers, testing hyperparameters, new features, like dropout) to test new advances, how good they are in NLP tasks.
If you have any questions, feel free to use the issues page, or contact us: zseder@gmail.com; pajkossy.katalin@nytud.mta.hu


## Sequential tagging

Library is ready for pos and ner (or any other bieo1 tagged) training. `hunvec/datasets/prepare.py` is doing preprocessing.

Unfortunately because of a pylearn2/theano reason, only `batch_size==1` can be used, but the training is working. There is also another script for evaluating (F-score and per-word precision) and for tagging with these models. Models can be read and training can be continued. Library supports featurizing, now only 3gram features, casing and gazetteer features are implemented, but it is simply extendable with pure python methods. 
There are many training options for `hunvec/seqtag/trainer.py`, see its help message.

Good to know: `IndexSequenceSpace.make_theano_batch()` and `VectorSequenceSpace.make_theano_batch()` in `pylearn2/space/__init__.py` has to be modified right now.

instead of 
~~~~
if batch_size == 1:
    return tensor.matrix(name=name)
~~~~

one should use
~~~~
if batch_size == 1 or batch_size is None:
    return tensor.matrix(name=name, dtype=self._dtype)
~~~~

## Sample calls:

Datasets has to be in the common format:
- one token per line
- empty line separates sentences
- in one line: <word> <tab> <tag>

Preparing dataset with train/test/devel split (before preparing, features can be turned on and off in `features/features.py`):
~~~~
python hunvec/datasets/prepare.py
-w 3
--test_file data/eng.bie1.test
--valid_file data/eng.bie1.devel
data/eng.bie1.train preprocessed_dataset.pickle
~~~~

For training and continuing a trained model with a given dataset:
~~~~
python hunvec/seqtag/trainer.py
--epochs 100
--regularization 1e-5
--valid_stop
--embedding 100
--hidden 200
--lr .1
--lr_lin_decay .1
--lr_scale
dataset.pickle
output_model
~~~~

For evaluation:
~~~~
python hunvec/seqtag/eval.py --fscore --sets test,valid,train dataset model
~~~~

For tagging:
~~~~
cut -f1 data/eng.bie1.train | python hunvec/seqtag/tagger.py model > tagged
~~~~

### multi-model training

There is support for using one models output in another model. One example use-case: train a pos tagger, and then use its output as input for a NE tagger.
To achieve this, one has to prepare datasets together in order to make words and features have the same index in both datasets. This is still experimental, we didn't achieve good results yet, but we are continuously working on it.

Example running for multiple preparing:
~~~~
python hunvec/datasets/prepare.py
-w 3
--test_file test1.tsv,test2.tsv
--valid_file valid1.tsv,valid2.tsv
train1.tsv,train2.tsv model1.pickle,model2.pickle
~~~~

Then, if a first model has been trained individually from the first dataset and saved into `model1.pickle`, it can be passed to `trainer.py` with option `--embedded_model`. Don't forget to use datasets that are prepared together.

## Language modeling

Dataset preprocessing, network creation and training is done, but without hierarchical softmax, or negative sampling, it is very slow. For negative sampling, there is ongoing work in the pylearn2 communitiy, there was https://github.com/lisa-lab/pylearn2/pull/1406, which wasn't finished, but hopefully soon will be.
