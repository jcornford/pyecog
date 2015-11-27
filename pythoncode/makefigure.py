import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import pickle

import utils as phd

cs = ['none','b','r','k','grey','grey']
treedata = np.loadtxt('../treedata500.csv',delimiter = ',')
#treedata = np.loadtxt('../knndata.csv',delimiter = ',')

print treedata.shape

figure = plt.figure(figsize = (12,8),facecolor='white')
grid =  gridspec.GridSpec(2, 2,)

classifier = pickle.load(open('../pca','rb'))
pca = classifier.pca_iss_features
ax1 = plt.subplot(grid[0])

for i in range(pca.shape[0]):
    ax1.scatter(pca[i,0], pca[i,1], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k', s = 30)

ax1.set_xlabel("Principle Component 1",)
ax1.set_ylabel("Principle Component 2", )
var = sum((classifier.pca.explained_variance_ratio_))
ax1.set_title("2d PCA projection: {:.2%} variance explained".format(var))

ax2 = plt.subplot(grid[1])
image = mpimg.imread('../5cv.png')
ax2.imshow(image)
ax2.axis('off')
ax2.set_title('5-fold cross validation')

ax3 = plt.subplot(grid[2])
n_trees = treedata[:,0]
ax3.errorbar(n_trees, treedata[:, 1],yerr = treedata[:,2]/np.sqrt(5), label ='Cross validation', color = phd.mc['k'])
#ax3.plot(n_trees, treedata[:, 1],color = phd.mc['k'], label ='Cross validation')
ax3.plot(n_trees, treedata[:, 3], color = phd.mc['r'], label ='Test dataset')

ax3.legend(frameon = False, loc ='best', fontsize = 10)

xlim = ax3.get_xlim()
#ax3.hlines(0.25, 0,xlim[1])
myxlim = (-0.1, xlim[1])
ax3.set_xlim(myxlim)

ylim = ax3.get_ylim()
myylim = (ylim[0],1)
ax3.set_ylim(myylim)
#ax3.set_ylim(0,1)

ax3.set_xlabel('Number of Trees')
ax3.set_ylabel('Classifier performance')
ax3.set_title('Classifier performance')

import pandas as pd
df = pd.read_pickle('../feature_importance')
ax4 = plt.subplot(grid[3])
ax4.set_title('Feature importance')
ax4.set_ylabel('Importance (%)')
df = df.sort(columns=0)
df.plot(kind='bar', ax = ax4,rot = 80, legend = False, grid = False,
        color=phd.mc['k'],)



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
plt.tight_layout()
plt.show()
#plt.savefig('v01.png')

