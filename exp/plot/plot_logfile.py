import sys
from matplotlib import plot as plt



def proc_and_plot(to_plot_fns, toplot_score, toplot_dataset):
    return [1,2,3], [95, 96, 97], 'ro', [1,2,3], [96, 97, 98], 'bo'

def proc_and_plot(to_plot_fns, toplot_score, toplot_dataset):
    colours = ['ro', 'bo', 'go', 'yo']
    plt(proc(to_plot_fns, toplot_score, toplot_dataset))




def main():

    toplot_score = ','.split(sys.argv[1])
    toplot_dataset = ','.split(sys.argv[2])
    to_plot_fns = sys.argv[3:]
    proc_and_plot(to_plot_fns, toplot_score, toplot_dataset)

if __name__ == "__main__":
    main()
