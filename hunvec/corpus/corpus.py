#import logging
import sys
from random import shuffle
import cPickle

import numpy
import tables

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.space import IndexSpace, CompositeSpace

from hunvec.utils.binary_tree import BinaryTreeEncoder


def create_hdf5_file(fn, X_shape, y_shape, num_labels):
    h5file = tables.openFile(fn, mode="w", title="Dataset")
    filters = tables.Filters(complib='blosc', complevel=5)
    gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
    h5file.createCArray(gcolumns, 'X', atom=tables.Int32Atom(), shape=X_shape,
                        title="Data_X", filters=filters)
    h5file.createCArray(gcolumns, 'y', atom=tables.Int32Atom(), shape=y_shape,
                        title="Data_y", filters=filters)
    h5file.createCArray(gcolumns, 'num_labels', atom=tables.Int32Atom(),
                        shape=(1,), title="num_labels", filters=filters)
    node = h5file.getNode('/', 'Data')
    node.num_labels[0] = num_labels
    h5file.flush()
    return h5file


def insert_arrays_to_hdf5(h5f, X, y, start_pos):
    node = h5f.getNode('/', 'Data')
    node.X[start_pos:start_pos+X.shape[0], :] = X
    node.y[start_pos:start_pos+X.shape[0], :] = y


class Corpus(object):
    def __init__(self, dump_path, corpus_fn=None, window_size=3, top_n=10000,
                 hs=False, future=False):
        self.ws = window_size
        self.top_n = top_n
        self.skip_str = "<unk>"
        self.hs = hs
        self.future = future
        self.corpus_fn = corpus_fn
        self.dump_path = dump_path
        if corpus_fn is not None:
            self.compute_needed_words(corpus_fn)
            if hs:
                self.w_enc = BinaryTreeEncoder(self.needed).word_encoder
            self.corpus_f = open(self.corpus_fn)
            self.num_examples = sum(len(_) for _ in self.read_corpus())
            del self.corpus_f

    def compute_needed_words(self, fn):
        v = {}
        for l in open(fn):
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                v[w] = v.get(w, 0) + 1
        sorted_v = sorted(v.iteritems(), key=lambda x: -x[1])
        needed = sorted_v[:self.top_n]
        self.index2word = dict((i, w) for i, (w, f) in enumerate(needed))
        self.index2word[-1] = self.skip_str
        self.vocab = dict((k, i) for i, (k, _) in enumerate(needed))
        needed = dict((self.vocab[w], f) for w, f in needed)
        needed[-1] = sum(v for _, v in sorted_v[self.top_n:])
        self.needed = needed

    def read_corpus(self):
        for l in self.corpus_f:
            l = l.decode("utf-8")
            s = l.split()
            s = [(self.vocab[w] if w in self.vocab else -1) for w in s]
            res = []
            for ngr, y in self.sentence_to_examples(s):
                if y == -1:
                    continue
                if self.hs:
                    y = self.w_enc(y)
                else:
                    y = [y]
                res.append((ngr, y))
            yield res

    def sentence_to_examples(self, s):
        n = self.ws
        end = (len(s) - n if not self.future else
               len(s) - 2 * n)
        for i in xrange(end):
            if self.future:
                context = s[i:i+n] + s[i+n+1:i+n+1+n]
            else:
                context = s[i:i+n]
            y = s[i+n]
            yield context, y

    def get_dump_filenames(self):
        tr = "{0}.train.h5".format(self.dump_path)
        tst = "{0}.test.h5".format(self.dump_path)
        v = "{0}.valid.h5".format(self.dump_path)
        i2w_fn = "{0}.i2w.pickle".format(self.dump_path)
        return tr, tst, v, i2w_fn

    def read_batch(self, batch=10000):
        examples = [], []
        if hasattr(self, 'leftover'):
            for a, b in self.leftover:
                examples[0].append(a)
                examples[1].append(b)
            del self.leftover

        try:
            for sen_res in self.read_corpus():
                for i, (a, b) in enumerate(sen_res):
                    examples[0].append(a)
                    examples[1].append(b)
                    if len(examples[0]) == batch:
                        self.leftover = sen_res[i+1:]
                        return examples
        except StopIteration:
            pass
        return examples

    def create_dump_files(self, ratios=[.7, .15, .15], batch=100000):
        self.corpus_f = open(self.corpus_fn)
        tr_fn, tst_fn, v_fn, i2w_fn = self.get_dump_filenames()
        ws = (self.ws * 2 if self.future else self.ws)
        sizes = [int(s * self.num_examples) + 1 for s in ratios]
        y_shape1 = (len(self.needed) - 1 if self.hs else 1)
        tr_X_shape = (sizes[0], ws)
        tr_y_shape = (sizes[0], y_shape1)
        tst_X_shape = (sizes[1], ws)
        tst_y_shape = (sizes[1], y_shape1)
        v_X_shape = (sizes[2], ws)
        v_y_shape = (sizes[2], y_shape1)
        trf = create_hdf5_file(tr_fn, tr_X_shape, tr_y_shape, len(self.needed))
        tsf = create_hdf5_file(tst_fn, tst_X_shape, tst_y_shape,
                               len(self.needed))
        vf = create_hdf5_file(v_fn, v_X_shape, v_y_shape, len(self.needed))
        cPickle.dump(self.index2word, open(i2w_fn, 'wb'), -1)

        bsizes = [int(s * batch) for s in ratios]
        bc = 0
        while True:
            examples = self.read_batch(batch)
            if len(examples[0]) == 0:
                break
            y = numpy.array(examples[1], dtype=numpy.int8)
            X = numpy.array(examples[0])
            total = len(y)
            indices = range(total)
            shuffle(indices)
            training = bsizes[0]
            test = bsizes[1]
            valid = bsizes[2]
            if total < batch:
                training = int(total * ratios[0])
                valid = int(total * ratios[2])
            training_indices = indices[:training]
            valid_indices = indices[training:training + valid]
            test_indices = indices[training+valid:training+valid+test]

            train_X = X[training_indices, :]
            train_y = y[training_indices]
            test_X = X[test_indices, :]
            test_y = y[test_indices]
            valid_X = X[valid_indices, :]
            valid_y = y[valid_indices]

            tr_start = bc * bsizes[0]
            tst_start = bc * bsizes[1]
            v_start = bc * bsizes[2]
            insert_arrays_to_hdf5(trf, train_X, train_y, tr_start)
            insert_arrays_to_hdf5(tsf, test_X, test_y, tst_start)
            insert_arrays_to_hdf5(vf, valid_X, valid_y, v_start)
            bc += 1

    def read_hdf5_to_dataset(self, fn):
        self.h5file = tables.openFile(fn, mode='r')
        data = self.h5file.getNode('/', "Data")
        dataset = DenseDesignMatrixPyTables(X=data.X, y=data.y)
        return dataset, data.num_labels[0]

    def set_spaces(self, dataset, dim_X, dim_Y, m):
        dataset.X_space = IndexSpace(dim=dim_X, max_labels=m)
        dataset.y_space = IndexSpace(dim=dim_Y, max_labels=m)
        X_source = 'features'
        y_source = 'targets'
        space = CompositeSpace((dataset.X_space, dataset.y_space))
        source = (X_source, y_source)
        dataset._iter_data_specs = (dataset.X_space, 'features')
        dataset.data_specs = (space, source)

    def read_dataset(self):
        tr_fn, tst_fn, v_fn, i2w_fn = self.get_dump_filenames()
        trd, m = self.read_hdf5_to_dataset(tr_fn)
        tstd, _ = self.read_hdf5_to_dataset(tst_fn)
        vd, _ = self.read_hdf5_to_dataset(v_fn)
        dim_X = trd.X.shape[1]
        dim_Y = (1 if len(trd.y.shape) == 1 else trd.y.shape[1])
        self.set_spaces(trd, dim_X, dim_Y, m)
        self.set_spaces(tstd, dim_X, dim_Y, m)
        self.set_spaces(vd, dim_X, dim_Y, m)
        self.index2word = cPickle.load(open(i2w_fn))
        return {'train': trd, 'test': tstd, 'valid': vd}, m


def main():
    c = Corpus(corpus_fn=sys.argv[1], dump_path=sys.argv[2],
               window_size=5, top_n=20000, hs=False, future=False)
    c.create_dump_files()

if __name__ == "__main__":
    main()
