###targets:

- Language Modeling (based on Bengio2003)
- sequential tagging (based on Collobert et al 2011)

###Done

## Language modeling

Dataset preprocessing, network creation and training is done, but without hierarchical softmax, or negative sampling, it is very slow. For negative sampling, there is ongoing work in the pylearn2 communitiy, there was https://github.com/lisa-lab/pylearn2/pull/1406, which wasn't finished, but hopefully soon will be.

## Sequential tagging

Library is ready for pos and ner (or any other bieo1 tagged) training. `hunvec/datasets/word_tagger_dataset.py` is doing preprocessing.

Unfortunately because of a pylearn2/theano reason, only `batch_size==1` can be used, but the training is working still. There is also another script for evaluating (F-score and per-word precision) and for tagging with these models. Models can be read and training can be continued. Library supports featurizing, now only 3gram features, casing and gazetteer features are implemented. There are many options for `hunvec/seqtag/trainer.py`.

Good to know: `IndexSequenceSpace.make_theano_batch()` in `pylearn2/space/__init__.py` has to be modified right now.

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
