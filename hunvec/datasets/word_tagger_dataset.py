import functools
import argparse
import cPickle

import numpy

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class

from hunvec.utils.data_splitter import datasplit, shuffled_indices
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer


def read_vocab(fn, lower=True, decoder='utf-8'):
    d = {}
    for l in open(fn):
        w = l.strip()
        if decoder:
            w = w.decode(decoder)
        if lower:
            w = w.lower()
        d[w] = len(d)
    return d


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
        ws = (window_size * 2 + 1)
        self.total_feats = total_feats * ws
        self.feat_num = feat_num * ws 
        self.n_classes = n_classes
        space = CompositeSpace((
            IndexSequenceSpace(max_labels=vocab_size, dim=ws),
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
    def create_from_tagged_corpus(c, window_size=3, pad_num=-2):
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

            if len(words) < 3:
                continue

            cwords.append(lwords)
            cfeatures.append(lfeats)
            y.append(ltags)

        return cwords, cfeatures, y, vocab, classes


def init_presplitted_corpus(args):
    ws = args.window
    train_fn = args.train_file
    valid_fn = args.valid_file
    test_fn = args.test_file
    featurizer = Featurizer()
    w2i = (read_vocab(args.vocab) if args.vocab else None)
    train_c = TaggedCorpus(train_fn, featurizer, w2i=w2i)
    valid_c = TaggedCorpus(valid_fn, featurizer, w2i=train_c.w2i,
                           t2i=train_c.t2i)
    test_c = TaggedCorpus(test_fn, featurizer, w2i=valid_c.w2i,
                          t2i=valid_c.t2i)
    train_res = WordTaggerDataset.create_from_tagged_corpus(
        train_c, window_size=ws)
    valid_res = WordTaggerDataset.create_from_tagged_corpus(
        valid_c, window_size=ws)
    test_res = WordTaggerDataset.create_from_tagged_corpus(
        test_c, window_size=ws)
    words, feats, y, _, _ = train_res
    n_words = len(train_res[3] | test_res[3] | valid_res[3])
    n_classes = len(train_res[4] | test_res[4] | valid_res[4])
    train_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                 featurizer.total, featurizer.feat_num,
                                 n_classes)
    words, feats, y, _, _ = valid_res
    valid_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                 featurizer.total, featurizer.feat_num,
                                 n_classes)
    words, feats, y, _, _ = test_res
    test_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                featurizer.total, featurizer.feat_num,
                                n_classes)
    d = {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
    return d, train_c


def init_split_corpus(args):
    ws = args.window
    featurizer = Featurizer()
    w2i = (read_vocab(args.vocab) if args.vocab else None)
    c = TaggedCorpus(args.train_file, featurizer, w2i=w2i)
    res = WordTaggerDataset.create_from_tagged_corpus(c, window_size=ws)
    words, feats, y, vocab, classes = res
    n_words, n_classes = len(vocab), len(classes)
    d = create_splitted_datasets(words, feats, y, args.train_split, n_words,
                                 ws, featurizer.total, featurizer.feat_num,
                                 n_classes)
    return d, c


def load_dataset(fn):
    d, c = cPickle.load(open(fn))
    return d, c


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('train_file')
    argparser.add_argument('output_file')
    argparser.add_argument('--test_file')
    argparser.add_argument('--valid_file')
    argparser.add_argument('--train_split', default=[0.8, 0.1, 0.1],
                           help='train/test/valid ratios, used when only' +
                           ' training file is given')
    argparser.add_argument('-w', '--window', default=5, type=int,
                           dest='window')
    argparser.add_argument('--vocab', dest='vocab', help='add vocab file to ' +
                           'predefine what words will be used. Useful if ' +
                           'later external word vectors will be used, so ' +
                           'network needs to be prepared for words that are' +
                           ' not in the training data')
    return argparser.parse_args()


def main():
    args = create_argparser()
    if args.test_file and args.valid_file:
        res = init_presplitted_corpus(args)
    else:
        # use train_split option
        res = init_split_corpus(args)
    with open(args.output_file, 'wb') as of:
        cPickle.dump(res, of, -1)


if __name__ == "__main__":
    main()
