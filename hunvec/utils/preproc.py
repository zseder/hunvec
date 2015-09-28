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
        self.vocab = {}    
            
    def compile_num_regex(self):
        self.num_regex = re.compile('^[0-9\-\.]+$')
    
    def get_vocab(self, training_f):
        training_fh = open(training_f)
        counts = self.get_word_count(training_fh)
        training_fh.close()
        self.vocab = set(filter(lambda x:counts[x] > 
            self.cutoff, counts.keys()))

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


    def write_replaced_text(self, f):
        fh = open(f)
        fh_replaced = open('{}_num_{}.cutoff_{}'.format(
            f, self.num, self.cutoff), 'w')
        for l in fh:
            l = l.strip('\n').decode('utf-8')
            if len(l) == 0:
                fh_replaced.write('\n')
                continue
            w, t = l.split('\t')
            if self.num and self.num_regex.match(w) is not None:
                w = '__NUM__'
            elif  self.rare and w.lower() not in self.vocab:
                w = '__RARE__'
            fh_replaced.write(u'{}\t{}\n'.format(w, t).encode('utf-8'))    

    def create_replaced_files(self, training_f, devel_f, test_f):
        if self.rare:
            self.get_vocab(training_f)
        self.write_replaced_text(training_f)
        self.write_replaced_text(devel_f)
        self.write_replaced_text(test_f)
        



def main():
    
    training_f = sys.argv[1]
    devel_f = sys.argv[2]
    test_f = sys.argv[3]
    a = TrainingPreprocesser(rare=False)
    a.create_replaced_files(training_f, devel_f, test_f)

if __name__ == '__main__':
    main()
