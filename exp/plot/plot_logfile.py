import sys
from matplotlib import pyplot as plt
from numpy import array

def parse_result(l, toplot_score):
    if toplot_score == 'precision':
        return float(l[1:-1])
    if toplot_score == 'fscore':
        return float(l.split(',')[-1].split(')')[0].strip())

def generate_scores(fh, toplot_score):
    test = True
    for l in fh:
        l = l.strip('\n')
        if len(l) < 4:
            epoch = int(l)
            continue
        if l[0].isupper():
            test = True
            continue
        if test:
            score = parse_result(l, toplot_score)
            yield epoch, score
            test = False

def proc(fn, toplot_score):
    fh = open(fn)
    epochs = []
    scores = []
    prev_score = 0
    for e, sc in generate_scores(fh, toplot_score):
        if sc > prev_score:
            epochs.append(e)
            scores.append(sc)
            prev_score = sc
    return epochs, scores
  
def adjust(epoch_list, score_list):
    maxl = max([len(l) for l in epoch_list])
    for i in range(len(epoch_list)):
        e = epoch_list[i][-1]
        s = score_list[i][-1]
        for j in range(maxl-len(epoch_list[i])):
            epoch_list[i].append(e)
            score_list[i].append(s)
        
    return epoch_list, score_list 

def proc_fns(fns, toplot_score):
    epoch_list = []
    score_list = []        
    for fn in fns:
        e, sc = proc(fn, toplot_score)
        epoch_list.append(e)
        score_list.append(sc)
    epoch_list, score_list = adjust(epoch_list, score_list)    
    return epoch_list, score_list
    


def proc_and_plot(to_plot_fns, toplot_score):
      values = []
      values = proc_fns(to_plot_fns, toplot_score)
      plt.plot(array(values[0]).T, array(values[1]).T)
      plt.show()
  
  
  
  
def main():
  
      toplot_score = sys.argv[1]
      to_plot_fns = sys.argv[2:]
      proc_and_plot(to_plot_fns, toplot_score)
  
if __name__ == "__main__":
      main()

