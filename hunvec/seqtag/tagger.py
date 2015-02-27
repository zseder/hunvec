import argparse

from pylearn2.utils import serial

from hunvec.utils.fscore import FScCounter
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.seqtag.word_tagger_dataset import WordTaggerDataset


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('input')
    argparser.add_argument('model')
    argparser.add_argument('--fscore', action='store_true',
                           help='if given, don\'t tag, only compute f1 score')
    return argparser.parse_args()


def load_and_score(args):
    wt = serial.load(args.model)
    c = TaggedCorpus(args.input, featurizer=wt.featurizer,
                     w2i=wt.w2i, t2i=wt.t2i)
    data = WordTaggerDataset.create_from_tagged_corpus(
        c, window_size=wt.window_size)
    words, feats, y, _, _ = data
    ds = WordTaggerDataset((words, feats), y, wt.vocab_size, wt.window_size,
                           wt.featurizer.total, wt.featurizer.feat_num,
                           wt.n_classes)
    wt.prepare_tagging()
    wt.f1c = FScCounter(c.i2t)
    print list(wt.get_score(ds, 'f1'))


def main():
    args = create_argparser()
    if args.fscore:
        load_and_score(args)
    else:
        raise Exception("Tagging is not implemented yet")


if __name__ == '__main__':
    main()
