import argparse
import cPickle

from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.datasets.tools import read_vocab, create_splitted_datasets
from hunvec.datasets.word_tagger_dataset import WordTaggerDataset


def prepare_presplitted_corpus(args, featurizer, w2i):
    ws = args.window
    train_fn = args.train_file
    valid_fn = args.valid_file
    test_fn = args.test_file
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
    n_words = len(train_c.w2i)
    return train_res, valid_res, test_res, n_classes, n_words, ws, train_c


def init_presplitted_corpus(args):
    featurizer = Featurizer()
    w2i = (read_vocab(args.vocab) if args.vocab else None)
    r = prepare_presplitted_corpus(args, featurizer, w2i)
    train_res, valid_res, test_res, n_classes, n_words, ws, train_c = r
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
    return d, train_c


def init_split_corpus(args):
    ws = args.window
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
    argparser.add_argument('--extra_inputs', help='extra input files for ' +
                           'presplitted mode when multiple corporas has to ' +
                           'be featurized together to reuse indices; ' +
                           'comma separates extra corpora, semicolon ' +
                           'separates train,test,valid splits inside one ' +
                           'corpora')
    argparser.add_argument('--extra_outputs', help='extra output files for ' +
                           'presplitted mode when multiple corporas has to ' +
                           'be featurized together to reuse indices; ' +
                           'comma separated values, same length as ' +
                           'extra_inputs')
    raise Exception("extra_inputs and extra_outputs handling has to be implemented")
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
