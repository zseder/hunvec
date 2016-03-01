from hunvec.corpus.raw_corpus import RawCorpus

class TaggedCorpus(RawCorpus):

    def __init__(self, fn, featurizer=None, w2i=None, t2i=None,
            use_unknown=False, num=False, use_unknown_tags=False):
        RawCorpus.__init__(self, fn, featurizer, w2i,
                use_unknown, num)
        self.t2i = ({} if t2i is None else t2i)
        self.use_unknown_tags = use_unknown_tags

    def add_ints(self, sen):
        RawCorpus.add_ints(self, sen)
        for i in xrange(len(sen)):
            new_ti = (self.unk if self.use_unknown_tags else len(self.t2i))
            sen[i][1] = self.t2i.setdefault(sen[i][1], new_ti)

    def read(self, pre=False):
        for s in RawCorpus.read(self, pre, needed_fields=[0, 1]):
            yield s
