import sys
from collections import defaultdict
import re

class TrainingPreprocesser:

    def __init__(self, num=True, rare=True, cutoff=1):
        self.rare = rare
        self.num = num
        self.cutoff=cutoff
        if self.num:
            self.compile_num_regex()
            
    def compile_num_regex(self):
        self.num_regex = re.compile('^[0-9\-\.]+$')
    
    def get_vocab(self, training_fh):
        counts = self.get_word_count(training_fh)
        return set(filter(lambda x:counts[x] > self.cutoff, counts.keys()))

    def get_word_count(self, training_fh):
        word_counts = defaultdict(int)
        for l in training_fh:
            l = l.strip('\n').decode('utf-8').lower()
            if len(l) == 0:
                continue
            w, t = l.split('\t')
            if self.num and self.num_regex.match(w) is not None:
                w = '__NUM__'
            word_counts[w] += 1
        return word_counts    


    def write_replaced_text(self, f, vocab):
        fh = open(f)
        fh_replaced = open('{}_num_{}.cutoff_{}'.format(
            f, self.num, self.cutoff), 'w')
        for l in fh:
            l = l.strip('\n').decode('utf-8').lower()
            if len(l) == 0:
                fh_replaced.write('\n')
                continue
            w, t = l.split('\t')
            if self.num and self.num_regex.match(w) is not None:
                w = '__NUM__'
            if w in vocab:
                fh_replaced.write(u'{}\t{}\n'.format(w, t).encode('utf-8'))
                continue
            fh_replaced.write(u'__RARE__\t{}\n'.format(w, t).encode('utf-8'))

    def create_replaced_files(self, training_f, devel_f, test_f):
        training_fh = open(training_f)
        vocab = self.get_vocab(training_fh) 
        training_fh.close()
        self.write_replaced_text(training_f, vocab)
        self.write_replaced_text(devel_f, vocab)
        self.write_replaced_text(test_f, vocab)
        



def main():
    
    training_f = sys.argv[1]
    devel_f = sys.argv[2]
    test_f = sys.argv[3]
    a = TrainingPreprocesser()
    a.create_replaced_files(training_f, devel_f, test_f)

if __name__ == '__main__':
    main()
