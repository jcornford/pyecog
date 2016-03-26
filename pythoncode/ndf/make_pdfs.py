'''

This needs to be made parallel...!

'''
from __future__ import print_function
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
dir = os.path.dirname(__file__)
filename = os.path.join(dir, '../HelveticaNeue-Light.otf')
prop = fm.FontProperties(fname=filename)

def plot_traces(to_plot,
                start = 0,
                labels = None,
                savepath = None,
                filename = None,
                format_string = ".pdf",
                prob_thresholds = None, verbose = False):
    '''

    Args:
        to_plot: expecting this with
        start:
        labels: if None, traces are all assigned with 1's
        savestring:
        format_string:
        prob_thresholds: if not None, used to color code the index (0-sure- "k", 1-unsure "r")

    Returns:

    '''

    if labels is None:
        labels = np.ones(to_plot.shape[0])
    if prob_thresholds is None:
        prob_thresholds = np.zeros(to_plot.shape[0]).astype(int)

    colors = ['b','r','g','k']
    colors = ['k','r','g','b','purple']

    print('Plotting traces...')
    for section in range(int(np.ceil(to_plot.shape[0]/40.0))):


        fig = plt.figure(figsize=(8.27, 11.69), dpi=20)
        plt.axis('off')
        plt.title('Traces '+ str((section+start)*40)+ ':' + str(((section+start)+1)*40)+'  1:black 2:red 3:green 4:blue', fontproperties=prop, fontsize = 14)
        time = np.linspace(1,10,to_plot.shape[1])
        annotation_colors = ['k','r']

        if section == to_plot.shape[0]/40:
            n_plots = to_plot.shape[0]%40
            if verbose:
                print(str(section*40)+ ' : ' + str(((section)*40)+n_plots))
        else:
            n_plots = 40
            if verbose:
                print(str(section*40)+ ' : ' + str((section+1)*40)+',')

        for i in [ii + (section)*40 for ii in range(n_plots)]:
            ax = fig.add_subplot(20,2,(i%40)+1)
            ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
            ax.annotate(str(i+start), xy = (0,0.3), fontsize = 10, color = annotation_colors[prob_thresholds[i]], fontproperties=prop)

            ax.axis('off')
            ax.set_xlim((0,10))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_yaxis().tick_left()
            ax.get_xaxis().tick_bottom()

        if savepath:
            plt.savefig(os.path.join(savepath,filename+'_'+str(section+ (start+1)/40)+format_string))
            #plt.savefig(os.path.join(savepath,filename+'_'+str(section)+format_string))

        else:
            plt.show()

        plt.close('all')


def plot_traces_hdf5(to_plot,
                labels = None,
                savepath = None,
                filename = None,
                format_string = ".pdf",
                prob_thresholds = None,
                trace_len_sec = 5,
                verbose = False):

    if not format_string.startswith('.'):
        format_string = '.'+format_string

    if labels is None:
        labels = np.ones(to_plot.shape[0])
    if prob_thresholds is None:
        prob_thresholds = np.zeros(to_plot.shape[0]).astype(int)

    colors = ['k','r','g','b','purple']
    print('Plotting traces...')
    for section in range(int(np.ceil(to_plot.shape[0]/40.0))):

        fig = plt.figure(figsize=(8.27, 11.69), dpi=20)
        plt.axis('off')
        plt.title('Seconds '+ str(section*40*trace_len_sec)+ ':' + str((section+1)*40*trace_len_sec), fontsize = 14,fontproperties=prop)
        time = np.linspace(1,10,to_plot.shape[1])
        annotation_colors = ['k','r']

        if section == to_plot.shape[0]/40:
            n_plots = to_plot.shape[0]%40
            if verbose:
                print(str(section*40)+ ' : ' + str(((section)*40)+n_plots))
        else:
            n_plots = 40
            if verbose:
                print(str(section*40)+ ' : ' + str((section+1)*40)+',')

        for i in [ii + (section)*40 for ii in range(n_plots)]:
            ax = fig.add_subplot(20,2,(i%40)+1)
            ax.annotate(str(i), xy = (0,0.5), fontsize = 10,color = 'black', fontproperties=prop)

            ax.plot(time, to_plot[i,:], color = colors[int(labels[i])], linewidth = 0.5)
            ax.set_title(str(i*trace_len_sec), fontsize = 8, fontproperties=prop)
            ax.axis('off')
            ax.set_xlim((0,10))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_yaxis().tick_left()
            ax.get_xaxis().tick_bottom()

        if savepath:
            plt.savefig(os.path.join(savepath,filename+'_'+str(section)+format_string))

        plt.close('all')
    print('Done')



if __name__ == "__main__":
    training_tuple = pickle.load(open('../training_label_traces_tuple','rb'))
    training_tuple = pickle.load(open('../validation_label_traces_tuple','rb'))

    labels = training_tuple[0]
    data = training_tuple[1]
    print('plotting '+ str(data.shape[0])+ 'traces')
    plot_traces(data,labels,savestring='../validation ',format_string='.pdf')