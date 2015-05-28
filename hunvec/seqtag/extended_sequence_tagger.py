import theano

from hunvec.seqtag.sequence_tagger import SequenceTaggerNetwork
from hunvec.seqtag.extended_hidden_network import ExtendedHiddenNetwork
from hunvec.datasets.extended_word_tagger_dataset import \
        ExtendedWordTaggerDataset

class ExtendedSequenceTaggerNetwork(SequenceTaggerNetwork):
    def __init__(self, embedded_model, *args, **kwargs):
        self.embedded_model = embedded_model
        super(ExtendedSequenceTaggerNetwork, self).__init__(*args, **kwargs)

    def __getstate__(self):
        d = super(ExtendedSequenceTaggerNetwork, self).__getstate__()
        d['embedded_model'] = self.embedded_model
        return d

    def _create_tagger(self):
        self.tagger = ExtendedHiddenNetwork(
            self.embedded_model.n_classes,
            self.vocab_size, self.window_size, self.total_feats,
            self.feat_num, self.hdims, self.edim, self.fedim, self.n_classes)

    def set_dataset(self, data):
        new_data = {}
        for k, d in data.iteritems():
            new_data[k] = ExtendedWordTaggerDataset(self.embedded_model, d)
        self._create_data_specs(new_data['train'])
        self.dataset = new_data

    def prepare_tagging(self):
        X = self.get_input_space().make_theano_batch(batch_size=1)
        Y = self.fprop(X)
        self.f = theano.function([X[0], X[1], X[2]], Y)
        self.start = self.A.get_value()[0]
        self.end = self.A.get_value()[1]
        self.A_value = self.A.get_value()[2:]

    def process_input(self, words, feats):
        inner_tags = self.embedded_model.tag_sen(
            words, feats, return_probs=True)
        return self.f(words, feats, inner_tags)
