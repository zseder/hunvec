import sys
from matplotlib import pyplot as plt
from numpy import array
 
 
def proc(to_plot_fns, toplot_score, toplot_dataset):
     return [[1,2,3],[1,2,3]],[[95, 96, 97],[40, 41, 42]]
  
def proc_and_plot(to_plot_fns, toplot_score, toplot_dataset):
      values = []
      values = proc(to_plot_fns, toplot_score, toplot_dataset)
      plt.plot(array(values[0]).T, array(values[1]).T)
      plt.show()
  
  
  
  
def main():
  
      toplot_score = ','.split(sys.argv[1])
      toplot_dataset = ','.split(sys.argv[2])
      to_plot_fns = sys.argv[3:]
      proc_and_plot(to_plot_fns, toplot_score, toplot_dataset)
  
if __name__ == "__main__":
      main()

