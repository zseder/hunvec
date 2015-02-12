class TaggedCorpus(object):
    def __init__(self, fn, featurizer=None, w2i=None, t2i=None):
        self.fn = fn
        self.featurizer = featurizer
        self.read()
        if featurizer is not None:
            self.add_features()

        self.w2i = w2i
        self.t2i = t2i

        self.turn_to_ints()

    def add_features(self):
        if self.featurizer is not None:
            self.featurizer.preprocess_corpus(self.corpus)
        for sen_i in xrange(len(self.corpus)):
            sen = self.corpus[sen_i]
            new_sen = [[w, t, self.featurizer.featurize(w)] for w, t in sen]
            self.corpus[sen_i] = new_sen

    def read(self):
        d = []
        s = []
        for l in open(self.fn):
            le = l.strip().split("\t")
            if len(le) == 2:
                w, pos = le[0], le[1]
                s.append([w, pos])
            else:
                d.append(s)
                s = []
        if len(s) > 0:
            d.append(s)
        self.corpus = d

    def turn_to_ints(self):
        w2i = ({} if self.w2i is None else self.w2i)
        t2i = ({} if self.t2i is None else self.t2i)
        for sen_i in xrange(len(self.corpus)):
            sen = self.corpus[sen_i]
            for i in xrange(len(sen)):
                sen[i][0] = w2i.setdefault(sen[i][0].lower(), len(w2i))
                sen[i][1] = t2i.setdefault(sen[i][1], len(t2i))

        self.t2i = t2i
        self.w2i = w2i
        self.i2t = [w for w, i in
                    sorted(self.t2i.iteritems(), key=lambda x: x[1])]
        self.i2w = [w for w, i in
                    sorted(self.w2i.iteritems(), key=lambda x: x[1])]
