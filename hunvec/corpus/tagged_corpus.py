from hunvec.corpus.raw_corpus import RawCorpus

class TaggedCorpus(RawCorpus):

    def __init__(self, fn, featurizer=None, w2i=None, t2i=None,
            use_unknown=False):
        RawCorpus.__init__(self, fn, featurizer, w2i, t2i, 
                use_unknown)

    def add_ints(self, sen):
        RawCorpus.add_ints(self, sen)
        for i in xrange(len(sen)):
            new_ti = (self.unk if self.use_unknown else len(self.t2i))
            sen[i][1] = self.t2i.setdefault(sen[i][1], new_ti)

    def read(self, pre=False):
        for s in RawCorpus.read(self, pre, needed_fields=[0, 1]):
            yield s
