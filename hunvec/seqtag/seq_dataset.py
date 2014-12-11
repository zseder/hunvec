import functools

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import SequentialSubsetIterator


class SeqDataset(Dataset):
    def __init__(self, X, y):
        super(SeqDataset, self).__init__()
        self.X = X
        self.y = y
        print X, y
        # TODO compute from X any y
        space = CompositeSpace([
            IndexSpace(max_labels=10, dim=3),
            IndexSpace(max_labels=2, dim=3)
        ])
        source = ('inputs', 'targets')
        self.data_specs = (space, source)

    def get_num_examples(self):
        return len(self.X)

    def get_data_specs(self):
        return self.data_specs

    def get_data(self):
        return self.X, self.y

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=1,
                 rng=None, data_specs=None, return_tuple=False):
        mode = SequentialSubsetIterator
        return FiniteDatasetIterator(
            self,
            mode(len(self.X), batch_size, num_batches, rng),
            data_specs=self.data_specs,
        )
