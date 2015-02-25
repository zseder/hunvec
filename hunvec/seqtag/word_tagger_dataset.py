import functools

import numpy

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class

from hunvec.utils.data_splitter import datasplit, shuffled_indices


def create_splitted_datasets(wa, fa, ya, ratios,
                             vocab_size, window_size, total_feats, feat_num,
                             n_classes):
    indices = shuffled_indices(len(wa), ratios)
    wa_train, wa_test, wa_valid = datasplit(wa, indices, ratios)
    fa_train, fa_test, fa_valid = datasplit(fa, indices, ratios)
    ya_train, ya_test, ya_valid = datasplit(ya, indices, ratios)
    kwargs = {
        "vocab_size": vocab_size,
        "window_size": window_size,
        "total_feats": total_feats,
        "feat_num": feat_num,
        "n_classes": n_classes,
    }
    d = {
        'train': WordTaggerDataset((wa_train, fa_train), ya_train,
                                   **kwargs),
        'test': WordTaggerDataset((wa_test, fa_test), ya_test,
                                  **kwargs),
        'valid': WordTaggerDataset((wa_valid, fa_valid), ya_valid,
                                   **kwargs)
    }
    return d


class WordTaggerDataset(Dataset):
    def __init__(self, X, y, vocab_size, window_size, total_feats, feat_num,
                 n_classes):
        super(WordTaggerDataset, self).__init__()
        self.X1 = X[0]
        self.X2 = X[1]
        self.y = y
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.total_feats = total_feats * window_size
        self.feat_num = feat_num * window_size
        self.n_classes = n_classes
        space = CompositeSpace((
            IndexSequenceSpace(max_labels=vocab_size, dim=window_size),
            IndexSequenceSpace(max_labels=self.total_feats,
                               dim=self.feat_num),
            IndexSequenceSpace(dim=1, max_labels=n_classes)
        ))
        source = ('words', 'features', 'targets')
        self.data_specs = (space, source)

    def get_num_examples(self):
        return len(self.X1)

    def get_data_specs(self):
        return self.data_specs

    def get_monitoring_data_specs(self):
        return self.data_specs

    def get_data(self):
        return self.X1, self.X2, self.y

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):

        if num_batches is None:
            if 'shuffle' in mode:
                num_batches = len(self.X1) / (batch_size)
            else:
                num_batches = len(self.X1) / batch_size

        mode = resolve_iterator_class(mode)
        if data_specs is None:
            data_specs = self.data_specs

        i = FiniteDatasetIterator(
            self,
            mode(len(self.X1), batch_size, num_batches, rng),
            data_specs=data_specs, return_tuple=return_tuple,
        )
        return i

    @staticmethod
    def create_from_tagged_corpus(c, window_size=3, pad_num=-1):
        words = []
        features = []
        y = []
        pad = [(pad_num, pad_num, pad_num)] * (window_size - 1)
        # include the word itself
        vocab, classes = set(), set()
        for sen in c.corpus:
            sen = list(pad) + sen + list(pad)
            sen_words, sen_features, sen_y = [], [], []
            # don't create data where y is pad
            for word_i in xrange(window_size - 1, len(sen) - window_size + 1):
                tag = sen[word_i][1]

                # the word is there, too
                window = [w for w, _, _ in
                          sen[word_i - window_size + 1: word_i + 1]]

                # combine together features for indices
                fs = []
                r = range(word_i - window_size + 1, word_i + 1)
                for mul, i in enumerate(r):
                    feats = sen[i][2]
                    if feats == pad_num:
                        feats = c.featurizer.fake_features()

                    # copy features to not change sentence data
                    feats = list(feats)
                    for feat_i in xrange(len(feats)):
                        feats[feat_i] += mul * c.featurizer.total
                    fs += feats

                sen_words.append(window)
                sen_features.append(fs)
                sen_y.append([tag])

                # counting
                vocab |= set(window)
                classes.add(tag)

            if len(sen_words) < 3:
                continue

            words.append(numpy.array(sen_words))
            features.append(numpy.array(sen_features))
            y.append(numpy.array(sen_y))

        return words, features, y, vocab, classes
