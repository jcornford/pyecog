import pickle

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from pythoncode import utils as phd

cs = ['none','b','r','k','grey','grey']
treedata = np.loadtxt('../treedata500.csv',delimiter = ',')
#treedata = np.loadtxt('../knndata.csv',delimiter = ',')

print treedata.shape

figure = plt.figure(figsize = (11,12),facecolor='white')
grid =  gridspec.GridSpec(3, 2,)

classifier = pickle.load(open('../saved_clf','rb'))

########### PCA plot ###############
ax1 = plt.subplot(grid[0])
pca = classifier.pca_iss_features
for i in range(pca.shape[0]):
    ax1.scatter(pca[i,0], pca[i,1], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k', s = 30)

ax1.set_xlabel("Principle Component 1",)
ax1.set_ylabel("Principle Component 2", )
var = sum((classifier.pca.explained_variance_ratio_))
ax1.set_title("2d PCA projection: {:.2%} variance explained".format(var))

######### LDA plot ###############
ax2 = plt.subplot(grid[1])
lda = classifier.lda_iss_features
for i in range(pca.shape[0]):
    ax2.scatter(lda[i,0], lda[i,1], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k', s = 30)

ax2.set_xlabel("LD 1",)
ax2.set_ylabel("LD 3", )

ax2.set_title('2d LDA projection')

######### Cross val plot ###########
ax3 = plt.subplot(grid[2])
image = mpimg.imread('../5cv.png')
ax3.imshow(image)
ax3.axis('off')
ax3.set_title('5-fold cross validation')

########### RF characterisation ########
ax4 = plt.subplot(grid[3])
n_trees = classifier.treedata[:,0]
ax4.errorbar(n_trees, classifier.treedata[:, 1], yerr =classifier.treedata[:, 2] / np.sqrt(5), label ='Cross validation 22d', color = phd.mc['k'])
#ax4.plot(n_trees, treedata[:, 1],color = phd.mc['k'], label ='Cross validation')
ax4.plot(n_trees, classifier.treedata[:, 3], color = phd.mc['r'], label ='Test dataset 22d')

ax4.legend(frameon = False, loc ='best', fontsize = 10)

xlim = ax4.get_xlim()
#ax4.hlines(0.25, 0,xlim[1])
myxlim = (-0.1, xlim[1])
ax4.set_xlim(myxlim)

ylim = ax4.get_ylim()
myylim = (ylim[0],1)
ax4.set_ylim(myylim)
#ax4.set_ylim(0,1)

ax4.set_xlabel('Number of Trees')
ax4.set_ylabel('Classifier performance')
ax4.set_title('Classifier performance')

########### Feature importance ##############
ax5 = plt.subplot(grid[4])
ax5.set_title('Feature importance')
ax5.set_ylabel('Importance (%)')
import pandas as pd
df = pd.DataFrame(classifier.r_forest.feature_importances_*100,classifier.feature_labels)
df = df.sort(columns=0)
df.plot(kind='bar', ax = ax5, rot = 80, legend = False, grid = False,
        color=phd.mc['k'], )



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

#ax3 = plt.subplot(grid[1])
plt.tight_layout()
plt.show()
#plt.savefig('v01.png')

