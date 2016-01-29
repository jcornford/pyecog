import pickle

import matplotlib.pyplot as plt
import numpy as np

from pythoncode import utils


def plot_traces(to_plot, labels = None, savestring = None, format_string = ".pdf"):
    if labels == None:
        labels = np.ones(to_plot.shape[0])
    colors = ['b','r','g','k','purple']
    for section in range(int(np.ceil(to_plot.shape[0]/40.0))):
        print str(section*40)+ ' : ' + str((section+1)*40)

        fig = plt.figure(figsize=(8.27, 11.69), dpi=20)
        plt.axis('off')
        plt.title('Traces '+ str(section*40)+ ':' + str((section+1)*40)+'  1:blue 2:red 3:green B1:black')
        mi = np.min(to_plot[section*40:(section+1)*40,:])
        mx = np.max(to_plot[section*40:(section+1)*40,:])
        time = np.linspace(1,10,5120)

        try:
            for i in range(40):
                ax = fig.add_subplot(20,2,i+1)
                i += (section)*40
                ax.axis('off')
                ax.plot(time,to_plot[i,:], color = colors[int(labels[i])-1], linewidth = 0.5)
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10)
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
                ax.annotate(str(i), xy = (0,0.3), fontsize = 10)
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
        if savestring:
            plt.savefig(savestring+str(section)+format_string)

if __name__ == "__main__":
    training_tuple = pickle.load(open('../training_label_traces_tuple','rb'))
    training_tuple = pickle.load(open('../validation_label_traces_tuple','rb'))

    labels = training_tuple[0]
    data = training_tuple[1]
    print 'plotting ',data.shape[0], 'traces'
    plot_traces(data,labels,savestring='../validation ',format_string='.pdf')