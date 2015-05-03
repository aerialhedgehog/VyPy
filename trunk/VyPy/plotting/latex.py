
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from math import sqrt

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # from http://nipunbatra.github.io/2014/08/latexify/
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 4.0 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        #golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        #fig_height = fig_width*golden_mean # height in inches
        fig_height = 3.5

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

preconfigure = latexify

SPINE_COLOR = 'gray'
def format_axes(ax = None):
    
    if ax is None:
        ax = plt.gca()
    
    #for spine in ['top', 'right']:
        #ax.spines[spine].set_visible(False)

    for spine in ['top', 'right','left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    #plt.tight_layout()

    return ax

def format_legend_box(lg=None):
    if lg is None:
        lg = plt.gca().get_legend()
    lg.get_frame().set_linewidth(0.5)
    lg.get_frame().set_edgecolor('gray')