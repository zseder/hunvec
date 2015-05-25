from math import sqrt

from hunvec.seqtag.word_tagger import WordTaggerNetwork
from pylearn2.models.mlp import CompositeLayer, Tanh


class ExtendedHiddenTagger(WordTaggerNetwork):
    def __init__(self, extender, *args, **kwargs):
        self.extender = extender
        super(ExtendedHiddenTagger, self).__init__(*args, **kwargs)

    def create_hidden_layers(self):
        # for parameter settings, see Remark 7 (Tricks) in NLP from scratch
        hiddens = []
        for i, hdim in enumerate(self.hdims):
            sc = 1. / hdim
            h = Tanh(layer_name='h{}'.format(i), dim=hdim,
                     istdev=1./sqrt(hdim), W_lr_scale=sc, b_lr_scale=sc)
            if i == 0:
                cl = CompositeLayer(
                    layer_name='eh0',
                    layers=[
                        h,
                        self.extender
                    ]
                )
                hiddens.append(cl)
                continue

            hiddens.append(h)
        return hiddens
