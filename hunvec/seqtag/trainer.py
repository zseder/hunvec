import sys
import argparse

from hunvec.seqtag.word_tagger_dataset import WordTaggerDataset
from hunvec.seqtag.word_tagger_dataset import create_splitted_datasets
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.seqtag.sequence_tagger import SequenceTaggerNetwork


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('train_file')


def init_network_corpus():
    fn = sys.argv[1]
    ws = 6
    featurizer = Featurizer()
    c = TaggedCorpus(fn, featurizer)
    res = WordTaggerDataset.create_from_tagged_corpus(c, window_size=ws)
    words, feats, y, vocab, classes = res
    n_words, n_classes = len(vocab), len(classes)
    d = create_splitted_datasets(words, feats, y, [.8, .1, .1], n_words,
                                 ws, featurizer.total, featurizer.feat_num,
                                 n_classes)
    wt = SequenceTaggerNetwork(vocab_size=d['train'].vocab_size,
                               window_size=d['train'].window_size,
                               total_feats=d['train'].total_feats,
                               feat_num=d['train'].feat_num,
                               n_classes=d['train'].n_classes,
                               edim=50, hdim=300, dataset=d['train'],
                               max_epochs=300)
    return c, d, wt


def init_network_presplitted_corpus():
    train_fn = sys.argv[1]
    valid_fn = sys.argv[2]
    test_fn = sys.argv[3]
    ws = 6
    featurizer = Featurizer()
    train_c = TaggedCorpus(train_fn, featurizer)
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
    wt = SequenceTaggerNetwork(vocab_size=d['train'].vocab_size,
                               window_size=d['train'].window_size,
                               total_feats=d['train'].total_feats,
                               feat_num=d['train'].feat_num,
                               n_classes=d['train'].n_classes,
                               edim=50, hdim=300, dataset=d['train'],
                               max_epochs=300)
    return d, wt, train_c, valid_c, test_c


def train_presplitted():
    d, wt, _, _, _ = init_network_presplitted_corpus()
    wt.create_algorithm(d, sys.argv[4])
    wt.train()


if __name__ == "__main__":
    train_presplitted()
    #load_and_score()
