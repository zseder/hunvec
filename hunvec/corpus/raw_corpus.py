from hunvec.datasets.tools import replace_numerals

class RawCorpus(object):
    
    def __init__(self, fn, featurizer=None, w2i=None,
                 use_unknown=False, num=False):
        self.fn = fn
        self.featurizer = featurizer
        self.use_unknown = use_unknown
        if featurizer is not None and not hasattr(featurizer, 'total'):
            self.featurizer.preprocess_corpus(self.read(pre=True))

        self.unk = -1
        self.w2i = ({} if w2i is None else w2i)
        self.num = num

    def add_features(self, sen):
        new_sen = []
        for word_data in sen:
            word_data.append(self.featurizer.featurize(word_data[0]))
            new_sen.append(word_data)
        return new_sen

    def replace_nums_in_sentence(self, s):
        new_sen = []
        for word_data in s:
            word_data[0] = replace_numerals(word_data[0])
            new_sen.append(word_data)
        return new_sen
        
    def read(self, pre=False, needed_fields=[0]):
        s = []
        for l in open(self.fn):
            l = l.decode('utf-8')
            if len(l.strip('\n')) == 0:
                if not pre:
                    s = self.add_features(s)
                    if self.num:
                        s = self.replace_nums_in_sentence(s)
                    self.add_ints(s)
                yield s
                s = []
                continue
            le = l.strip().split("\t")
            s.append([le[i] for i in filter(lambda x:x in needed_fields,
                xrange(len(le)))])
        if len(s) > 0:
            if not pre:
                s = self.add_features(s)
                if self.num:
                    s = self.replace_nums_in_sentence(s)
                self.add_ints(s)
            yield s

    def add_ints(self, sen):
        for i in xrange(len(sen)):
            new_wi = (self.unk if self.use_unknown else len(self.w2i))
            w = sen[i][0].lower()
            sen[i] = [self.w2i.get(w, new_wi)]\
                      + sen[i][1:] + [sen[i][0]]
            if not self.use_unknown and w not in self.w2i:
                self.w2i[w] = new_wi
 
