
import bokeh.plotting as bp
from bokeh.io import output_notebook
from .utils import filterArray
import numpy as np

def plot(data_dict, nsec = 3600):
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
