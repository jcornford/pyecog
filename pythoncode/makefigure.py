import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



treedata = np.loadtxt('../treedata.csv',delimiter = ',')
#treedata = np.loadtxt('../knndata.csv',delimiter = ',')

print treedata.shape

figure = plt.figure()
grid =  gridspec.GridSpec(2, 1,)

ax3 = plt.subplot(grid[0])
n_trees = treedata[:,0]
ax3.plot(n_trees, treedata[:, 1], label ='Training')
ax3.plot(n_trees, treedata[:, 2], label ='Test')
ax3.plot(n_trees, treedata[:, 3], label ='Validation')
#ax3.hlines(0.25,0,treedata.shape[0], linestyle = '--')
ax3.legend(frameon = False, loc ='best', fontsize = 10)
ax3.set_ylim(0.5, 1.01)
ax3.set_xlabel('Number of Trees')



def hide_spines():
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in plt._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

hide_spines()

#ax2 = plt.subplot(grid[1])

plt.show()

