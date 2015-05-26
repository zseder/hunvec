from hunvec.utils.data_splitter import datasplit, shuffled_indices
from hunvec.datasets.word_tagger_dataset import WordTaggerDataset

def read_vocab(fn, lower=True, decoder='utf-8'):
    d = {}
    for l in open(fn):
        w = l.strip()
        if decoder:
            w = w.decode(decoder)
        if lower:
            w = w.lower()
        if w in d:
            continue
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
