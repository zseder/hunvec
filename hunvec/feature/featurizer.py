from collections import defaultdict
from functools import partial


def fake_feat(word):
    return 'fake:'


def case_feature(word):
    if word.isupper():
        return 'case:upper'
    elif word.islower():
        return 'case:lower'
    elif word[0].isupper() and word[1:].islower():
        return 'case:capital'


def suffix_ngram_feature(word, n=3, end_index=None):
    w2 = ("^" * n) + word.lower() + "$"
    s = 'ngr_{}:'.format(end_index)
    if end_index is not None:
        return s + w2[-n+end_index:end_index]
    else:
        return s + w2[-n:]
snf = suffix_ngram_feature


def lasts(x):
    return snf(x, 3, None)


def last_but_ones(x):
    return snf(x, 3, -1)


def last_but_twos(x):
    return snf(x, 3, -2)


def gazetteer_feat(x, name='', set_=([])):
    if x.lower() in set_:
        return 'gaz_{0}:1'.format(name)
    else:
        return 'gaz_{0}:0'.format(name)

gaz_fns = {
    'languages': '/home/eszter/CIA_lists/misc/languages.txt',
    'religion': '/home/eszter/CIA_lists/misc/religion.txt',
    'ethnic': '/home/eszter/CIA_lists/misc/ethnic.txt',
    'nationality': '/home/eszter/CIA_lists/misc/nationality.txt',
    'capital': '/home/eszter/CIA_lists/loc/capital.txt',
    'city': '/home/eszter/CIA_lists/loc/city.txt',
    'country': '/home/eszter/CIA_lists/loc/countries.txt',
    'port': '/home/eszter/CIA_lists/loc/ports.txt',
    'region': '/home/eszter/CIA_lists/loc/region.txt',
    'org': '/home/eszter/CIA_lists/org/org.txt',
    'party': '/home/eszter/CIA_lists/org/party.txt',
    'person': '/home/eszter/freebase_lists/pers/per_all'
}


class Featurizer(object):
    def __init__(self, gazetteer_needed=False, fns=None):

        # HACK fake_feat has to be always there to avoid empty arrays
        self.feats = [
            #fake_feat,
            case_feature,
            lasts, last_but_ones, last_but_twos
        ]
        # HACK2 fake_feat messes up results (maybe slows down SGD) so it is 
        # removed when there are others as well
        if len(self.feats) > 1:
            self.feats = [f for f in self.feats if f is not fake_feat]

        if gazetteer_needed:
            if fns is None:
                fns = gaz_fns
            self.load_gazetteers(fns)
            for c in self.gazetteers:
                self.feats.append(
                    partial(gazetteer_feat, name=c, set_=self.gazetteers[c]))
        self.feat_num = len(self.feats)
        self.kept = [{} for _ in self.feats]

    def load_gazetteers(self, fns):
        self.gazetteers = {}
        for c in fns:
            g = []
            for l in open(fns[c]):
                for i in l.strip().decode('utf-8').lower().split():
                    g.append(i)
            self.gazetteers[c] = set(g)

    def preprocess_corpus(self, corpus):
        """ reads the whole corpus to preprocess features for detecting
        feature numbers. For example how many starting trigrams or ending
        trigrams there are"""
        feature_counters = [defaultdict(int) for _ in self.feats]
        for sentence in corpus:
            for word in sentence:
                for i, feat in enumerate(self.feats):
                    w_feat = feat(word[0])
                    if w_feat is None:
                        continue
                    feature_counters[i][w_feat] += 1
        self.keep_features(feature_counters)

    def keep_features(self, feat_counts, min_count=5):
        for feat_i in xrange(len(self.feats)):
            for feat, feat_c in feat_counts[feat_i].iteritems():
                if feat_c >= min_count:
                    if feat not in self.kept[feat_i]:
                        self.kept[feat_i][feat] = len(self.kept[feat_i])
        self.build_final_data()

    def build_final_data(self):
        # using k + 1 because of "else" or "fake" features (needed)
        self.total = sum(len(k) + 1 for k in self.kept)
        self.fake_features = []
        if self.total == 0:
            return

        # list with fake features, needed for pads
        self.fake_features.append(len(self.kept[0]))

        # reverse dict for easier printouts
        self.i2f = [None] * self.total

        for k in self.kept[0].iterkeys():
            self.i2f[self.kept[0][k]] = k

        # increasing values in dictionaries to have unique numbers for features
        shift = 0
        for i in xrange(1, len(self.kept)):
            shift += len(self.kept[i-1]) + 1
            for key in self.kept[i].keys():
                self.i2f[self.kept[i][key] + shift] = key
                self.kept[i][key] = self.kept[i][key] + shift
            self.fake_features.append(
                self.fake_features[-1] + len(self.kept[i]) + 1)

    def featurize(self, word):
        all_w_feats = []
        for feat_i, feat in enumerate(self.feats):
            f = feat(word)
            if f in self.kept[feat_i]:
                loc = self.kept[feat_i][f]
            else:
                loc = self.fake_features[feat_i]
            all_w_feats.append(loc)

        return all_w_feats
