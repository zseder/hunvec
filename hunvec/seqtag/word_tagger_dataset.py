import functools

import numpy

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.iteration import SequentialSubsetIterator

from hunvec.utils.data_splitter import datasplit


def create_splitted_datasets(wa, fa, ya, ratios):
        vocab_size = wa.max() - wa.min() + 1
        feat_num = fa.max() - fa.min() + 1
        n_classes = ya.max() - ya.min() + 1
        wa_train, wa_test, wa_valid = datasplit(wa, ratios)
        fa_train, fa_test, fa_valid = datasplit(fa, ratios)
        ya_train, ya_test, ya_valid = datasplit(ya, ratios)
        kwargs = {
            "vocab_size": vocab_size,
            "window_size": wa_train.shape[1],
            "feat_num": feat_num,
            "n_classes": n_classes,
        }
        return {
            'train': WordTaggerDataset((wa_train, fa_train), ya_train,
                                       **kwargs),
            'test': WordTaggerDataset((wa_test, fa_test), ya_test,
                                      **kwargs),
            'valid': WordTaggerDataset((wa_valid, fa_valid), ya_valid,
                                       **kwargs)
        }


class WordTaggerDataset(Dataset):
    def __init__(self, X, y, vocab_size, window_size, feat_num, n_classes,
                 iteration_mode=None):
        super(WordTaggerDataset, self).__init__()
        self.X1 = X[0]
        self.X2 = X[1]
        self.y = y
        space = CompositeSpace((
            IndexSequenceSpace(max_labels=vocab_size, dim=window_size),
            IndexSequenceSpace(max_labels=feat_num, dim=window_size),
            IndexSequenceSpace(max_labels=n_classes, dim=1)
        ))
        source = ('inputs', 'features', 'targets')
        self.data_specs = (space, source)
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.feat_num = feat_num
        self.n_classes = n_classes
        if iteration_mode is None:
            self.iteration_mode = SequentialSubsetIterator
        else:
            self.iteration_mode = iteration_mode

    def get_num_examples(self):
        return len(self.X1)

    def get_data_specs(self):
        return self.data_specs

    def get_data(self):
        return self.X1, self.X2, self.y

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):

        if num_batches is None:
            num_batches = len(self.X1) / (batch_size * 10)

        mode = resolve_iterator_class(mode)

        i = FiniteDatasetIterator(
            self,
            mode(len(self.X1), batch_size, num_batches, rng),
            data_specs=self.data_specs,
        )
        return i

    @staticmethod
    def create_from_tagged_corpus(c, window_size=3, pad_num=-1,
                                  ratios=[.8, .10, .10]):
        words = []
        features = []
        y = []
        pad = [(pad_num, pad_num)] * window_size
        # include the word itself
        fake_feats = [0] * (window_size + 1)
        for sen in c.corpus:
            sen = list(pad) + sen + list(pad)
            # don't create data where y is pad
            for word_i in xrange(window_size, len(sen) - window_size):
                tag = sen[word_i][1]

                # the word is there, too
                window = [w for w, p in sen[word_i - window_size: word_i + 1]]
                fs = fake_feats
                words.append(window)
                features.append(fs)
                y.append([tag])

        words_array = numpy.array(words)
        feats_array = numpy.array(features)
        y_array = numpy.array(y)
        return create_splitted_datasets(words_array,
                                        feats_array,
                                        y_array, ratios)
