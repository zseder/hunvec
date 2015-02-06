import numpy

class FScCounter:

    def __init__(self, labels):
        
        self.get_mappings(labels)
        self.init_confusion_matrix() 
    
    def get_mappings(self, labels):

        self.lab2ind = dict([(t, i) for i, t in enumerate(labels)])
        self.get_phrase_beginners()
        self.get_phrase_indeces() 

    def get_phrase_beginners(self):  # indeces of labels beginning a phrase-> {index:phrase}
        
        self.phrase_beginners = {}
        for label in self.lab2ind:
            if label != 'O':
                part, categ = label.split('-')
                if part == 'B' or part == '1':
                    ind = self.lab2ind[label]
                    self.phrase_beginners[ind] = {}
                    self.phrase_beginners[ind]['cat'] = categ
                    self.phrase_beginners[ind]['part'] = part

    def get_phrase_indeces(self):   # phrase continuing labels -> {phrase: indeces}
        
        self.phrase_indeces = {}
        for label in self.lab2ind:
            if label != 'O':
                part, categ = label.split('-')
                if part == 'I' or part == 'E':
                    if categ not in self.phrase_indeces:
                        self.phrase_indeces[categ] = {}
                    self.phrase_indeces[categ][part] = self.lab2ind[label]
    
    def init_confusion_matrix(self):
        
        self.confusion_matrix = {}
        for k in self.phrase_indeces:
            self.confusion_matrix[k] = {}
            self.confusion_matrix[k]['tp'] = float(0)
            self.confusion_matrix[k]['fp'] = float(0)
            self.confusion_matrix[k]['fn'] = float(0)
    
    def calculate_f(self):

        for k in self.confusion_matrix:
            tp = self.confusion_matrix[k]['tp']
            fp = self.confusion_matrix[k]['fp']
            fn = self.confusion_matrix[k]['fn']
            if not (tp + fp == 0):
                precision = tp/(tp + fp)
            else:    
                precision = 'N/A'
            if not (tp + fn == 0):
                recall = tp/(tp + fn)    
            else:    
                recall = 'N/A'
            if not(precision == 'N/A' or recall == 'N/A'):
                f_score = 2*precision*recall/(precision + recall)
            else:
                f_score = 'N/A'
            yield (k, precision, recall, f_score)    


    def generate_phrases(self, sen):

        len_ = len(sen)
        i = -1
        in_phrase = False
        categ = None
        ph_i = 0
        while i < len_-1:
            i += 1
            array = sen[i]
            ind = numpy.argmax(array)
            if ind in self.phrase_beginners:
                categ = self.phrase_beginners[ind]['cat']
                if self.phrase_beginners[ind]['part'] == '1':
                    in_phrase = False
                    yield (categ, i, i)
                else:
                    in_phrase = True
                    ph_i = i
                continue                 
            if in_phrase:
                if ind == self.phrase_indeces[categ]['E']:
                    in_phrase = False
                    yield (categ, ph_i, i)
                elif ind != self.phrase_indeces[categ]['I']:
                    in_phrase = False
                
    
    def count_score(self, gold, input_):

        for i, sen in enumerate(gold):
            self.process_sen(sen, input_[i])
            for sc in self.calculate_f():
                yield sc
    
    
    def process_sen(self, gold_sen, input_sen):

        gold_phrases = set([gp for gp in self.generate_phrases(gold_sen)])
        input_phrases = set([gp for gp in self.generate_phrases(input_sen)])
        self.update_scores(gold_phrases, input_phrases) 
    
    
    def update_scores(self, gold_phrases, input_phrases):
        
        for ph in gold_phrases.intersection(input_phrases):
            categ = ph[0]
            self.confusion_matrix[categ]['tp'] += 1
        for ph in gold_phrases.difference(input_phrases):
            categ = ph[0]
            self.confusion_matrix[categ]['fn'] += 1
        for ph in input_phrases.difference(gold_phrases):
            categ = ph[0]
            self.confusion_matrix[categ]['fp'] += 1



def main():

    list_of_tags = ['B-P', 'I-P', 'E-P', '1-P', 'O', 'B-L', 'I-L', 'E-L', 'O']
    gold = numpy.array([[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,0,1], [0,0,0,1,0]]])
    input_ = numpy.array([[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]])
    a = FScCounter(list_of_tags)
    for sc in a.count_score(gold, input_):
        print repr(sc)

if __name__ == "__main__":
    main()
