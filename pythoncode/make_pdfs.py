import stfio_plot as sp

def plot_traces(to_plot, labels = None, savestring = None):
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
                    sp.plot_scalebars(ax, linewidth=1.0, yunits='mV',div = 3.0, xunits= 's', sb_yoff = 0.1, sb_xoff = -0.1)

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
                    sp.plot_scalebars(ax, linewidth=1.0, yunits='mV',div = 3.0, xunits= 's',
                                     sb_yoff = 0.1, sb_xoff = -0.1)
        if savestring:
            plt.savefig(savestring+str(section)+".pdf")