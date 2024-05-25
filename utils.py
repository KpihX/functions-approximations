import matplotlib.pyplot as plt

""" Functions """

def set_fig(title = "Functions approximations", xlabel = "x", ylabel = "y"):         
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
