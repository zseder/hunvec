class TaggedCorpus(object):
    def __init__(self, fn, featurizer=None, w2i=None, t2i=None,
                 use_unknown=False):
        self.fn = fn
        self.featurizer = featurizer
        if featurizer is not None:
            self.featurizer.preprocess_corpus(self.read())

        self.w2i = ({} if w2i is None else w2i)
        self.t2i = ({} if t2i is None else t2i)
        self.use_unknown = use_unknown
        self.unk = -1

    def add_features(self, sen):
        new_sen = [[w, t, self.featurizer.featurize(w)] for w, t in sen]
        return new_sen

    def read_into_memory(self):
        self.corpus = list(self.read())

    def read(self):
        s = []
        for l in open(self.fn):
            le = l.strip().split("\t")
            if len(le) == 2:
                w, pos = le[0], le[1]
                s.append([w, pos])
            else:
                if self.featurizer:
                    s = self.turn_to_ints(self.add_features(s))
                yield s
                s = []
        if len(s) > 0:
            if self.featurizer:
                s = self.turn_to_ints(self.add_features(s))
            yield s

    def turn_to_ints(self, sen):
        for i in xrange(len(sen)):
            new_wi = (self.unk if self.use_unknown else len(self.w2i))
            new_ti = (self.unk if self.use_unknown else len(self.t2i))
            sen[i][0] = self.w2i.setdefault(sen[i][0].lower(), new_wi)
            sen[i][1] = self.t2i.setdefault(sen[i][1], new_ti)
