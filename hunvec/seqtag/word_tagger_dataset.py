import functools

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import SequentialSubsetIterator


class WordTaggerDataset(Dataset):
    def __init__(self, X, y):
        super(WordTaggerDataset, self).__init__()
        self.X1 = X[0]
        self.X2 = X[1]
        self.y = y
        # TODO compute from X any y
        space = CompositeSpace((
            IndexSpace(max_labels=10, dim=3),
            IndexSpace(max_labels=2, dim=3),
            IndexSpace(max_labels=2, dim=1)
        ))
        source = ('inputs', 'features', 'targets')
        self.data_specs = (space, source)

    def get_num_examples(self):
        return len(self.X1)

    def get_data_specs(self):
        return self.data_specs

    def get_data(self):
        return self.X1, self.X2, self.y

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=1,
                 rng=None, data_specs=None, return_tuple=False):
        mode = SequentialSubsetIterator
        i = FiniteDatasetIterator(
            self,
            mode(len(self.X1), batch_size, num_batches, rng),
            data_specs=self.data_specs,
        )
        return i

    @staticmethod
    def create_from_tagged_corpus(c, window_size=3, pad_num=-1):
        words = []
        features = []
        y = []
        pad = [pad_num] * window_size
        fake_feats = [0] * window_size
        for sen in c.corpus:
            sen = list(pad) + sen + list(pad)
            for word_i in xrange(window_size, len(sen)):
                w = sen[word_i]
                window = sen[word_i - window_size: word_i]
                fs = fake_feats
                words.append(window)
                features.append(fs)
                y.append([w])
        return WordTaggerDataset((words, features), y)
