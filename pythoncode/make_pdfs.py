'''

This needs to be made parallel...!

'''

import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from pythoncode import utils

prop = fm.FontProperties(fname='../HelveticaNeue-Light.otf')

def plot_traces(to_plot,
                labels = None,
                savepath = None,
                format_string = ".pdf",
                prob_thresholds = None):
    '''

    Args:
        to_plot: expecting this with
        labels: if None, traces are all assigned with 1's
        savestring:
        format_string:
        prob_thresholds: if not None, used to color code the index (0-sure- "k", 1-unsure "r")

    Returns:

    '''

    if labels == None:
        labels = np.ones(to_plot.shape[0])


    if prob_thresholds == None:
        prob_thresholds = np.zeros(to_plot.shape[0]).astype(int)

    colors = ['b','r','g','k','purple']
    for section in range(int(np.ceil(to_plot.shape[0]/40.0))):
        plt.close('all')
        print str(section*40)+ ' : ' + str((section+1)*40)

        fig = plt.figure(figsize=(8.27, 11.69), dpi=20)
        plt.axis('off')
        plt.title('Traces '+ str(section*40)+ ':' + str((section+1)*40)+'  1:blue 2:red 3:green 4:black', fontproperties=prop, fontsize = 14)
        mi = np.min(to_plot[section*40:(section+1)*40,:])
        mx = np.max(to_plot[section*40:(section+1)*40,:])

        time = np.linspace(1,10,to_plot.shape[1])

        annotation_colors = ['k','r']
        try:
            for i in range(40):
                ax = fig.add_subplot(20,2,i+1)
                i += (section)*40
                ax.axis('off')
                ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10, color = annotation_colors[prob_thresholds[i]], fontproperties=prop)
                ax.axis('off')
                ax.set_ylim((mi,mx))
                ax.set_xlim((0,10))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_yaxis().tick_left()
                ax.get_xaxis().tick_bottom()
                i -= (section)*40
                if i == 39:
                    utils.plot_scalebars(ax, linewidth=1.0, yunits='mV', div = 3.0, xunits='s', sb_yoff = 0.1, sb_xoff = -0.1)

        except IndexError:
            for i in range(to_plot.shape[0]%40):
                ax = fig.add_subplot(20,2,i+1)
                i += (section)*40
                ax.axis('off')
                ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10, color = annotation_colors[prob_thresholds[i]])
                ax.axis('off')
                ax.set_ylim((mi,mx))
                ax.set_xlim((0,10))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_yaxis().tick_left()
                ax.get_xaxis().tick_bottom()
                i -= (section)*40
                if i == 0:
                    utils.plot_scalebars(ax, linewidth=1.0, yunits='mV', div = 3.0, xunits='s',
                                         sb_yoff = 0.1, sb_xoff = -0.1)
        if savepath:
            plt.savefig(savepath+'_'+str(section)+format_string)

def plot_traces_hdf5(to_plot,
                labels = None,
                savestring = None,
                format_string = ".pdf",
                prob_thresholds = None,
                trace_len_sec = 5):
    '''

    Args:
        to_plot: expecting this with
        labels: if None, traces are all assigned with 1's
        savestring:
        format_string:
        prob_thresholds: if not None, used to color code the index (0-sure- "k", 1-unsure "r")

    Returns:

    '''

    if labels == None:
        labels = np.ones(to_plot.shape[0])


    if prob_thresholds == None:
        prob_thresholds = np.zeros(to_plot.shape[0]).astype(int)

    colors = ['k','r','g','b','purple']
    for section in range(int(np.ceil(to_plot.shape[0]/40.0))):
        print str(section*40)+ ' : ' + str((section+1)*40)

        fig = plt.figure(figsize=(8.27, 11.69), dpi=20)
        plt.axis('off')
        plt.title('Seconds '+ str(section*40*trace_len_sec)+ ':' + str((section+1)*40*trace_len_sec), fontsize = 14)
        mi = np.min(to_plot[section*40:(section+1)*40,:])
        mx = np.max(to_plot[section*40:(section+1)*40,:])

        time = np.linspace(1,10,to_plot.shape[1])

        annotation_colors = ['k','r']
        try:
            for i in range(40):
                ax = fig.add_subplot(20,2,i+1)
                i += (section)*40
                ax.axis('off')
                ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
                ax.set_title(str(i*trace_len_sec), fontsize = 6)
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10, color = annotation_colors[prob_thresholds[i]])
                ax.axis('off')
                #ax.set_ylim((mi,mx))
                ax.set_xlim((0,10))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_yaxis().tick_left()
                ax.get_xaxis().tick_bottom()
                i -= (section)*40
                #if i == 39:
                #    utils.plot_scalebars(ax, linewidth=1.0, yunits='mV', div = 3.0, xunits='s', sb_yoff = 0.1, sb_xoff = -0.1)

        except IndexError:
            for i in range(to_plot.shape[0]%40):
                ax = fig.add_subplot(20,2,i+1)
                i += (section)*40
                ax.axis('off')
                ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
                ax.set_title(str(i*trace_len_sec), fontsize = 6)
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10, color = annotation_colors[prob_thresholds[i]])
                ax.axis('off')
                #ax.set_ylim((mi,mx))
                ax.set_xlim((0,10))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_yaxis().tick_left()
                ax.get_xaxis().tick_bottom()
                i -= (section)*40
                #if i == 0:
                #    utils.plot_scalebars(ax, linewidth=1.0, yunits='mV', div = 3.0, xunits='s',
                 #                        sb_yoff = 0.1, sb_xoff = -0.1)
        if savestring:
            plt.savefig(savestring+str(section)+format_string)


if __name__ == "__main__":
    training_tuple = pickle.load(open('../training_label_traces_tuple','rb'))
    training_tuple = pickle.load(open('../validation_label_traces_tuple','rb'))

    labels = training_tuple[0]
    data = training_tuple[1]
    print 'plotting ',data.shape[0], 'traces'
    plot_traces(data,labels,savestring='../validation ',format_string='.pdf')