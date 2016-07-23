import sys
import bokeh.plotting as bp
if sys.version_info > (3,):
    from bokeh.io import output_notebook
else:
    from bokeh.plotting import output_notebook
from .utils import filterArray
import numpy as np
import matplotlib.pyplot as plt

def basic_plot(data_dict, time_tuple = (0,3600), arg_dict = {'color' :'k', 'linewidth':'1.5'}):
    """
    Inputs:
        - data dictionary (ndf[id] returns this) containing 'time' and 'data' keys,
        - time tuple in seconds, default (0,3600). Determines time period to plot
        - arg_dict: dictionary of parameters to pass to the plotting function
    """
    data = data_dict['data']
    time_ = data_dict['time']

    fig = plt.figure(figsize = (15, 4))
    ax = fig.add_subplot(111)
    #ax = fig.add_axes()
    indexes = np.logical_and(time_>time_tuple[0], time_<time_tuple[1])
    ax.plot(time_[indexes],data[indexes], **arg_dict)
    ax.set_xlim(time_tuple[0], time_tuple[1])
    return ax


def plot(data_dict, nsec = 3600):
    if sys.version_info < (3,):
        print('Plotting function is not available for python 2.7, please upgrade to 3!')
        sys.exit()
    print('WARNING: This plotting downsamples the data too much! ')
    data = data_dict['data']
    time = data_dict['time']

    x_downsamp = np.linspace(0,nsec,num = 50*nsec)
    y_downsamp = np.interp(x_downsamp, time, data)

    filtered = filterArray(y_downsamp, window_size=51, order = 3)


    output_notebook()

    p = bp.figure(webgl=True,
              plot_width=900,
              plot_height=500,)

    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Time (seconds)'
    p.yaxis.axis_label = 'Voltage'

    p.circle(x_downsamp, y_downsamp, size=4, legend='50hz interpolation',
          color='darkgrey', alpha=0.2)

    window_size = 30
    window = np.ones(window_size)/float(window_size)

    #data_avg = np.convolve(y_downsamp, window, 'valid')

    p.line(x_downsamp, filtered, legend='filtered', color='black')

    p.legend.location = "top_left"
    bp.show(p)
