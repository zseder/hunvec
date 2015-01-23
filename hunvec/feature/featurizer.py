from collections import defaultdict


def case_feature(word):
    if word.isupper():
        return 'case:upper'
    elif word.islower():
        return 'case:lower'
    elif word[0].isupper() and word[1:].islower():
        return 'case:capital'


def suffix_ngram_feature(word, n=3, end_index=None):
    w2 = ("^" * n) + word.lower() + "$"
    if end_index is not None:
        return w2[-n+end_index:end_index]
    else:
        return w2[-n:]
snf = suffix_ngram_feature


class Featurizer(object):
    def __init__(self):
        self.feats = [
            case_feature,
            #lambda x: snf(x, 3, None),  # last three (included $)
            #lambda x: snf(x, 3, -1),  # "last but one" three
            #lambda x: snf(x, 3, -2),
        ]
        self.feat_num = len(self.feats)

    def preprocess_corpus(self, corpus):
        """ reads the whole corpus to preprocess features for detecting
        feature numbers. For example how many starting trigrams or ending
        trigrams there are"""
        feature_counters = [defaultdict(int) for _ in self.feats]
        for sentence in corpus:
            for word, _ in sentence:
                for i, feat in enumerate(self.feats):
                    w_feat = feat(word)
                    if w_feat is None:
                        continue
                    feature_counters[i][w_feat] += 1
        self.keep_features(feature_counters)

    def keep_features(self, feat_counts, min_count=5):
        self.kept = [{} for _ in self.feats]
        for feat_i in xrange(len(self.feats)):
            for feat, feat_c in feat_counts[feat_i].iteritems():
                if feat_c >= min_count:
                    self.kept[feat_i][feat] = len(self.kept[feat_i])

        # using k + 1 because of "else" or "fake" features (needed)
        self.total = sum(len(k) + 1 for k in self.kept)
        self.feat_shifts = [0]
        for k in self.kept[:-1]:
            self.feat_shifts.append(len(k) + 1)

    def featurize(self, word):
        all_w_feats = []
        for feat_i, feat in enumerate(self.feats):
            f = feat(word)
            if f in self.kept[feat_i]:
                loc = self.feat_shifts[feat_i] + self.kept[feat_i][f]
            else:
                # not kept feature
                if feat_i + 1 < len(self.feat_shifts):
                    loc = self.feat_shifts[feat_i + 1] - 1
                else:
                    loc = self.total - 1
            all_w_feats.append(loc)

        return all_w_feats

    def fake_features(self):
        l = []
        for fs in self.feat_shifts[1:]:
            # the index before the shift index is the last (else) feature
            l.append(fs - 1)
        # else branch of last feature
        l.append(self.total - 1)
        return l
