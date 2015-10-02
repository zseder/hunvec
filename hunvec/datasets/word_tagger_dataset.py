import functools

import numpy

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.space import VectorSequenceSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class

class WordTaggerDataset(Dataset):
    def __init__(self, X, y, vocab_size, window_size, total_feats, feat_num,
                 n_classes):
        super(WordTaggerDataset, self).__init__()
        self.X1 = X[0]
        self.X2 = X[1]
        self.vocab_size = vocab_size + 1
        self.window_size = window_size
        ws = (window_size * 2 + 1)
        self.total_feats = total_feats * ws
        self.feat_num = feat_num * ws 
        self.n_classes = n_classes
        if len(y[0][0]) == 1:
            y = self.convert_to_sparse(y)
        self.y = y
        self._create_data_specs()

    def _create_data_specs(self):
        ws = (self.window_size * 2 + 1)
        space = CompositeSpace((
            IndexSequenceSpace(max_labels=self.vocab_size, dim=ws),
            IndexSequenceSpace(max_labels=self.total_feats,
                               dim=self.feat_num),
            VectorSequenceSpace(dim=self.n_classes)
        ))
        source = ('words', 'features', 'targets')
        self.data_specs = (space, source)

    def convert_to_sparse(self, y):
        ly = []
        for sen_y in y:
            a = numpy.zeros((len(sen_y), self.n_classes))
            for i in xrange(len(sen_y)):
                a[i, sen_y[i][0]] = 1.0
            ly.append(a)
        return ly

    def get_num_examples(self):
        return len(self.X1)

    def get_data_specs(self):
        return self.data_specs

    def get(self, source, next_index):
        # HACK since there are iterators that are 'fancy', and others are
        # not, we have to be prepared for numbered and sliced indexing
        if type(next_index) is slice:
            return (self.X1[next_index][0],
                    self.X2[next_index][0],
                    self.y[next_index][0])
        else:
            return self.X1[next_index], self.X2[next_index], self.y[next_index]

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):

        if num_batches is None:
            num_batches = len(self.X1) / (batch_size)

        mode = resolve_iterator_class(mode)
        i = FiniteDatasetIterator(
            self,
            mode(len(self.X1), batch_size, num_batches, rng),
            data_specs=data_specs,
        )
        return i

    @staticmethod
    def process_sentence(words, features, window_size, featurizer, pad_num=-2,
                         tags=None):
        pad = [pad_num] * window_size
        #pad = [pad_num] * (window_size - 1)

        # process words
        new_words = []
        words = pad + words + pad
        for word_i in xrange(window_size, len(words) - window_size):
            window_words = words[word_i - window_size:
                                 word_i + window_size + 1]
            new_words.append(window_words)
        new_words = numpy.array(new_words)

        # process features
        new_feats = []
        feats = pad + features + pad
        for feat_i in xrange(window_size, len(feats) - window_size):
            # combine together features for indices
            fs = []
            r = range(feat_i - window_size, feat_i + window_size + 1)
            for mul, i in enumerate(r):
                local_feats = feats[i]
                if local_feats == pad_num:
                    local_feats = featurizer.fake_features

                # copy features to not change sentence data
                local_feats = list(local_feats)
                for feat_i in xrange(len(local_feats)):
                    local_feats[feat_i] += mul * featurizer.total

                fs += local_feats
            new_feats.append(fs)
        new_feats = numpy.array(new_feats)
        res = [new_words, new_feats]

        if tags is not None:
            new_tags = numpy.array([[tag] for tag in tags])
            res.append(new_tags)

        return res

    @staticmethod
    def prepare_corpus(c, window_size=3, pad_num=-2):
        cwords = []
        cfeatures = []
        y = []
        # include the word itself
        vocab, classes = set(), set()
        for sen in c.read():
            words, tags, features, _ = [list(t) for t in zip(*sen)]

            res = WordTaggerDataset.process_sentence(
                words, features, window_size, c.featurizer, pad_num, tags)
            lwords, lfeats, ltags = res
            vocab |= set(words)
            classes |= set(tags)

            if len(words) < 1:
                continue

            cwords.append(lwords)
            cfeatures.append(lfeats)
            y.append(ltags)

        return cwords, cfeatures, y, vocab, classes
