import sys

from pylearn2.utils import serial

from hunvec.seqtag.trainer import init_network_presplitted_corpus
from hunvec.utils.fscore import FScCounter


def load_and_score():
    d, _, train_c, _, _ = init_network_presplitted_corpus()
    wt = serial.load(sys.argv[4])
    wt.prepare_tagging()
    wt.f1c = FScCounter(train_c.i2t)
    for k in d:
        data = d[k]
        print k, list(wt.get_score(data, 'f1'))
