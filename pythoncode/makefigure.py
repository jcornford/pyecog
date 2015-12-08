import pickle

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pythoncode import utils as phd

cs = ['none','b','r','k','grey','grey']
treedata = np.loadtxt('../treedata500.csv',delimiter = ',')
#treedata = np.loadtxt('../knndata.csv',delimiter = ',')

print treedata.shape

#figure = plt.figure(figsize=(8.27, 11.69),facecolor='white')
figure = plt.figure(figsize=(10, 12),facecolor='white')
grid =  gridspec.GridSpec(3, 2,)

classifier = pickle.load(open('../saved_clf','rb'))

########### PCA plot ###############
ax1 = plt.subplot(grid[0], projection = '3d')
pca = classifier.pca_iss_features
for i in range(pca.shape[0]):
    ax1.scatter(pca[i,0], pca[i,1],pca[i,2], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k',
                linewidth = 0.1,s = 20)

ax1.set_xlabel("PC 1",)
ax1.set_ylabel("PC 2", )
ax1.set_zlabel("PC 3", )
var = sum((classifier.pca.explained_variance_ratio_))
ax1.set_title("3d PCA projection: {:.2%} variance explained".format(var))
#ax1.view_init(elev=-144., azim=37)

######### LDA plot ###############
ax2 = plt.subplot(grid[1],projection='3d')
lda = classifier.lda_iss_features
for i in range(pca.shape[0]):
    ax2.scatter(lda[i,0], lda[i,1],lda[i,2], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k',
                s = 20,linewidth = 0.1, depthshade = True, alpha = 0.8)

ax2.set_xlabel("LD 1",)
ax2.set_ylabel("LD 2", )
ax2.set_zlabel("LD 3")
ax2.view_init(elev=-174., azim=73)
#ax2.set_zlim((-5,5))
#ax2.set_xlim((-4.5,6))
#ax2.set_ylim((-5,12))
ax2.set_title('3d LDA projection')

######### Cross val plot ###########
ax3 = plt.subplot(grid[2])
image = mpimg.imread('../5cv.png')
ax3.imshow(image)
ax3.axis('off')
ax3.set_title('5-fold cross validation')

########### RF characterisation ########
ax4 = plt.subplot(grid[3])
n_trees = classifier.treedata[:,0]

#ax4.errorbar(n_trees, classifier.treedata[:, 1], yerr =classifier.treedata[:, 2] / np.sqrt(5), label ='Cross validation 22d', color = phd.mc['k'])
#ax4.errorbar(n_trees, classifier.treedata[:, 4], yerr =classifier.treedata[:, 5] / np.sqrt(5), label ='Cross validation LDA 3d', color = phd.mc['grey'])
#ax4.errorbar(n_trees, classifier.treedata[:, 7], yerr =classifier.treedata[:, 8] / np.sqrt(5), label ='Cross validation PCA 3d', color = phd.mc['b'])
lw = 2
ax4.plot(n_trees, classifier.treedata[:, 1], color=phd.mc['k'],linestyle='-',linewidth = lw,  label='CV 22d')
ax4.plot(n_trees, classifier.treedata[:, 3], color=phd.mc['k'],linestyle='--', linewidth = lw, label='Test 22d')

ax4.plot(n_trees, classifier.treedata[:, 4], color=phd.mc['r'],linestyle='-', linewidth = lw,  label='CV LDA 3d')
ax4.plot(n_trees, classifier.treedata[:, 6], color = phd.mc['r'],linestyle = '--',linewidth = lw,  label ='Test LDA 3d')
print classifier.treedata[:,6]

ax4.plot(n_trees, classifier.treedata[:, 7], color = phd.mc['b'],linestyle = '-', linewidth = lw, label ='CV PCA 3d')
ax4.plot(n_trees, classifier.treedata[:, 9], color = phd.mc['b'],linestyle = '--', linewidth = lw, label ='Test PCA 3d')

xlim = ax4.get_xlim()
myxlim = (-0.1, xlim[1])
ax4.set_xlim(myxlim)

ax4.hlines(0.25, 0, xlim[1], label = 'Chance', linewidth = lw, color = phd.mc['grey'])

ax4.legend(frameon = False,ncol = 3, loc ='best', fontsize = 10)



ylim = ax4.get_ylim()
myylim = (ylim[0],1)
ax4.set_ylim(myylim)
ax4.set_ylim(0,1)

ax4.set_xlabel('Number of Trees')
ax4.set_ylabel('Classifier performance')
ax4.set_title('Classifier performance')

ax4.spines['right'].set_color('none')
ax4.spines['top'].set_color('none')
# Disable ticks.
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

########### Feature importance ##############
ax5 = plt.subplot(grid[4])
ax5.set_title('Feature importance')
ax5.set_ylabel('Importance (%)')
import pandas as pd
df = pd.DataFrame(classifier.r_forest.feature_importances_*100,classifier.feature_labels)
df = df.sort(columns=0)
df.plot(kind='bar', ax = ax5, rot = 80, legend = False, grid = False,
        color=phd.mc['k'], )

ax5.spines['right'].set_color('none')
ax5.spines['top'].set_color('none')
            # Disable ticks.
ax5.xaxis.set_ticks_position('bottom')
ax5.yaxis.set_ticks_position('left')

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

#hide_spines()

#ax3 = plt.subplot(grid[1])
plt.tight_layout()
plt.show()
#plt.savefig('v01.png')
'''
plt.figure()
ax2 = plt.subplot(111,projection='3d')
lda = classifier.lda_iss_features
for i in range(pca.shape[0]):
    ax2.scatter(lda[i,0], lda[i,1],lda[i,2], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k', s = 30,
                depthshade = True)

ax2.set_xlabel("LD 1",)
ax2.set_ylabel("LD 2", )
ax2.set_zlabel("LD 3")

ax2.set_title('2d LDA projection')
plt.show()
'''