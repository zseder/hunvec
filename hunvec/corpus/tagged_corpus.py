class TaggedCorpus(object):
    def __init__(self, fn):
        self.fn = fn
        self.read()

    def read(self):
        d = []
        s = []
        w2i = {}
        p2i = {}
        for l in open(self.fn):
            l = l.lower()
            le = l.strip().split("\t")
            if len(le) == 2:
                w, pos = le[0], le[1]
                w = w2i.setdefault(w, len(w2i))
                pos = p2i.setdefault(pos, len(p2i))
                s.append((w, pos))
            else:
                d.append(s)
                s = []
        if len(s) > 0:
            d.append(s)
        self.corpus = d
        self.p2i = p2i
        self.w2i = w2i
