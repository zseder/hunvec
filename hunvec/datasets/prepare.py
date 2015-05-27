import argparse
import cPickle

from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.datasets.tools import read_vocab, create_splitted_datasets
from hunvec.datasets.word_tagger_dataset import WordTaggerDataset


def prepare_presplitted_corpus(train_fn, valid_fn, test_fn, featurizer,
                               w2i, ws):
    train_c = TaggedCorpus(train_fn, featurizer, w2i=w2i)
    valid_c = TaggedCorpus(valid_fn, featurizer, w2i=train_c.w2i,
                           t2i=train_c.t2i, use_unknown=True)
    test_c = TaggedCorpus(test_fn, featurizer, w2i=valid_c.w2i,
                          t2i=valid_c.t2i, use_unknown=True)
    train_res = WordTaggerDataset.prepare_corpus(
        train_c, window_size=ws)
    valid_res = WordTaggerDataset.prepare_corpus(
        valid_c, window_size=ws)
    test_res = WordTaggerDataset.prepare_corpus(
        test_c, window_size=ws)
    n_classes = len(train_res[4] | test_res[4] | valid_res[4])
    return train_res, valid_res, test_res, n_classes, train_c


def init_presplitted_corpus(args):
    featurizer = Featurizer()
    w2i = (read_vocab(args.vocab) if args.vocab else None)
    res = []
    ws = args.ws
    # preprocess everything first, create datasets only after all preproc
    for i in xrange(len(args.train_file)):
        train_file = args.train_file[i]
        test_file = args.test_file[i]
        valid_file = args.valid_file[i]
        res.append(prepare_presplitted_corpus(
            train_file, test_file, valid_file, featurizer, w2i, args.ws))

    res2 = []
    for i in xrange(len(args.train_file)):
        train_res, valid_res, test_res, n_classes, train_c = res[i]
        n_words = len(train_c.w2i)
        words, feats, y, _, _ = train_res
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
        res2.append(d)
    return res2, train_c


def init_split_corpus(args):
    ws = args.ws
    featurizer = Featurizer()
    w2i = (read_vocab(args.vocab) if args.vocab else None)
    c = TaggedCorpus(args.train_file, featurizer, w2i=w2i)
    res = WordTaggerDataset.prepare_corpus(c, window_size=ws)
    words, feats, y, vocab, classes = res
    n_classes = len(classes)
    n_words = len(c.w2i)
    d = create_splitted_datasets(words, feats, y, args.train_split, n_words,
                                 ws, featurizer.total, featurizer.feat_num,
                                 n_classes)
    return d, c


def load_dataset(fn):
    d, c = cPickle.load(open(fn))
    return d, c


class CSL2L(argparse.Action):
    """ convert Comma Separated List into (2) single List"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('train_file', help='one or multiple filenames ' +
                          '(comma separated); if multiple corpora has to be ' +
                           'prepared (featurized) together, add all of them',
                           action=CSL2L)
    argparser.add_argument('output_file', help='output filename(s); same ' +
                           'length as train_files',
                           action=CSL2L)
    argparser.add_argument('--test_file', help='one or more filenames (see ' +
                           'train_file help msg',
                           action=CSL2L)
    argparser.add_argument('--valid_file', help='one or more filenames (see' +
                          ' train_file help msg',
                           action=CSL2L)
    argparser.add_argument('--train_split', default=[0.8, 0.1, 0.1],
                           help='train/test/valid ratios, used when only' +
                           ' training file is given')
    argparser.add_argument('-w', '--window', default=5, type=int,
                           dest='ws')
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
    
    for i in xrange(len(res[0])):
        with open(args.output_file[i], 'wb') as of:
            cPickle.dump((res[0][i], res[1]), of, -1)


if __name__ == "__main__":
    main()
