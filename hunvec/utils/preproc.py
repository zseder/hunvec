import sys
from collections import defaultdict
import re

class TrainingPreprocesser:

    def __init__(self, num=True, cutoff=None, list_fn=None):
        self.num = num
        self.cutoff = cutoff
        if self.num:
            self.compile_num_regex()
        self.list_fn = list_fn
        self.vocab = None
   
    def get_list_vocab(self):
        if self.list_fn != None:
            self.vocab = set([l.strip('\n').lower() for l in open(
                self.list_fn)])
            print len(self.vocab)
            
    def compile_num_regex(self):
        self.num_regex = re.compile('(^.*?)([0-9]+)(.*?$)')
    
    def get_cutoff_vocab(self, training_f):
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

        fh_replaced = open('{}_num_{}.cutoff_{}_dict_{}'.format(
            f, self.num, self.cutoff, self.list_fn.split('/')[-1]), 'w')
        for l in fh:
            numeric = False
            l = l.strip('\n').decode('utf-8')
            if len(l) == 0:
                fh_replaced.write('\n')
                continue
            w, t = l.split('\t')
            if self.num:
                m = self.num_regex.match(w)
                while m is not None:
                    numeric = True
                    w1, num, w2 = m.groups()
                    w = w1 + '__NUM__' + w2
                    m = self.num_regex.match(w)    
            if numeric == False and self.vocab != None and w.lower() not in self.vocab:
                if w.isupper():
                    w = 'RAREWORD'
                elif w[0].isupper() and w[1:].islower():
                    w = 'Rareword'
                else:
                     w = 'rareword'
            fh_replaced.write(u'{}\t{}\n'.format(w, t).encode('utf-8'))    

    def create_replaced_files(self, training_f, devel_f, test_f):
        if self.cutoff != None:
            self.get_cutoff_vocab(training_f)
        elif self.list_fn != None:
            self.get_list_vocab()
        self.write_replaced_text(training_f)
        self.write_replaced_text(devel_f)
        self.write_replaced_text(test_f)
        

def main():
    
    training_f = sys.argv[1]
    devel_f = sys.argv[2]
    test_f = sys.argv[3]
    a = TrainingPreprocesser(list_fn='/home/pajkossy/Proj/hunvec/embedding/wsj_150e_left_important')
    a.create_replaced_files(training_f, devel_f, test_f)

if __name__ == '__main__':
    main()
