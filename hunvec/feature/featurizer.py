import numpy


def case_feature(word):
    l = [0] * 4
    if word.isupper():
        l[0] = 1
    elif word.islower():
        l[1] = 1
    elif word[0].isupper() and word[1:].islower():
        l[2] = True
    else:
        l[3] = True
    return l


class Featurizer(object):
    def __init__(self):
        self.feats = [case_feature]

    def featurize(self, word):
        w_feats = []
        for feat in self.feats:
            f = feat(feat)
            w_feats += f
        return numpy.array(w_feats)
