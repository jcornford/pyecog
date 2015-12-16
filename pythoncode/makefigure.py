import pickle

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pythoncode import utils
from pythoncode import utils as phd

def main():
    cs = ['none','b','r','k','grey','grey']
    treedata = np.loadtxt('../treedata500.csv',delimiter = ',')
    #treedata = np.loadtxt('../knndata.csv',delimiter = ',')

    print treedata.shape

    #figure = plt.figure(figsize=(8.27, 11.69),facecolor='white')
    figure = plt.figure(figsize=(10, 12),facecolor='white')
    grid =  gridspec.GridSpec(3, 2,)

    classifier = pickle.load(open('../saved_clf','rb'))

    #pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    #classifier = pickle.load(open(pth,'rb'))


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
def lda_plot():
    pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    classifier = pickle.load(open('../saved_clf','rb'))
    cs = ['none','b','r','k','grey','grey']

    fig = plt.figure(figsize = (7,5))
    ax1 = fig.add_subplot(111,projection='3d')
    lda = classifier.lda_iss_features

    s1 = lda[classifier.labels==1]
    s2 = lda[classifier.labels==2]
    s3 = lda[classifier.labels==3]
    bl = lda[classifier.labels==4]

    size = 40
    alpha = 1.0
    lw = 0.1
    ax1.scatter(s1[:,0], s1[:,1],s1[:,2], c = utils.mc['k'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha, zorder = 2,
                depthshade = True)
    ax1.scatter(s2[:,0], s2[:,1],s2[:,2], c = utils.mc['r'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha,zorder = 2,
                depthshade = True)
    ax1.scatter(s3[:,0], s3[:,1],s3[:,2], c = utils.mc['b'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha,zorder = 2,
                depthshade = True)
    ax1.scatter(bl[:,0], bl[:,1],bl[:,2], c = utils.mc['grey'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha,zorder = 2,
                depthshade = True)

    ax1.set_xlabel("LD 1",)
    ax1.set_ylabel("LD 2",)
    ax1.set_zlabel("LD 3")
    ax1.view_init(elev=30., azim=-60)
    #ax1.view_init(elev=-60., azim=72)
    ax1.view_init(elev=-174., azim=73)
    #ax1.set_zlim((-5,5))
    #ax1.set_xlim((-4.5,6))
    #ax1.set_ylim((-5,12))
    ax1.set_title('3d Linear Discriminant analysis projection')
    plt.tight_layout()
    plt.savefig('/Users/jonathan/PhD/Presentations/lda.png')
    plt.show()

def pca_plot():
    pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    classifier = pickle.load(open('../saved_clf','rb'))
    cs = ['none','b','r','k','grey','grey']

    fig = plt.figure(figsize = (7,5))
    ax1 = fig.add_subplot(111,projection='3d')
    pca = classifier.pca_iss_features

    s1 = pca[classifier.labels==1]
    s2 = pca[classifier.labels==2]
    s3 = pca[classifier.labels==3]
    bl = pca[classifier.labels==4]

    size = 40
    alpha = 1.0
    lw = 0.1
    ax1.scatter(s1[:,0], s1[:,1],s1[:,2], c = utils.mc['k'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha, zorder = 3,
                depthshade = True)
    ax1.scatter(s2[:,0], s2[:,1],s2[:,2], c = utils.mc['r'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha,zorder = 2,
                depthshade = True)
    ax1.scatter(s3[:,0], s3[:,1],s3[:,2], c = utils.mc['b'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha,zorder = 1,
                depthshade = True)
    ax1.scatter(bl[:,0], bl[:,1],bl[:,2], c = utils.mc['grey'], edgecolor = 'k', s=size, linewidth=lw, alpha=alpha, zorder = 4,
                depthshade = True)

    ax1.set_xlabel("PC 1",)
    ax1.set_ylabel("PC 2",)
    ax1.set_zlabel("PC 3")
    ax1.view_init(elev=30., azim=-60.)
    ax1.view_init(elev=-174., azim=73)
    #ax1.view_init(elev=-144., azim=37)
    #ax2.view_init(elev=-174., azim=73)
    #ax1.view_init(elev=-162., azim=72)
    #ax1.set_zlim((-5,5))
    ax1.set_xlim((-6,8))
    #ax1.set_ylim((-5,12))
    var = sum((classifier.pca.explained_variance_ratio_))
    ax1.set_title("3d PCA projection: {:.2%} variance explained".format(var))
    plt.tight_layout()

    plt.savefig('/Users/jonathan/PhD/Presentations/pca.png')
    plt.show()


def scatter_plot_matrix():
    pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    classifier = pickle.load(open('../saved_clf','rb'))
    #cs = ['none','b','r','k','grey','grey']
    import seaborn as sns
    import pandas as pd
    sns.palplot(sns.color_palette("hls", 8))
    sns.color_palette("hls", 8)
    mc2 = [

       (0.14901960784313725, 0.13725490196078433, 0.13725490196078433),
       (0.8235294117647058, 0.34509803921568627, 0.34509803921568627),
       (0.30196078431372547, 0.4588235294117647, 0.7019607843137254),
       (0.7725490196078432, 0.7764705882352941, 0.7803921568627451),

       ]
    sns.palplot(mc2)

    X = classifier.iss_features[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]]
    fig = plt.figure(figsize=(8, 11))
    sns.set_palette(mc2)
    df = pd.DataFrame(X, columns =['Feature'+str(i+1) for i in range(X.shape[1])])
    print classifier.labels.shape
    df['Network States'] = classifier.labels
    pg = sns.PairGrid(df,
                  vars=['Feature'+str(i+1) for i in range(X.shape[1])],
                  hue="Network States",
                  size = 2,
                 )
    #pg = sns.pairplot(df)# hue="Network States")
    pg.map(plt.scatter)
    #pg.map_lower(plt.scatter)
    #pg.map_upper(plt.scatter)
    #pg.map_diag(plt.scatter)

    #plt.savefig(static+'scattermatrix.pdf')
    plt.show()

def perfomance():

    pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    classifier = pickle.load(open('../saved_clf','rb'))

    fig = plt.figure(figsize = (5.5,4.5))
    ax1 = fig.add_subplot(111)
    n_trees = classifier.treedata[:,0]

    #ax4.errorbar(n_trees, classifier.treedata[:, 1], yerr =classifier.treedata[:, 2] / np.sqrt(5), label ='Cross validation 22d', color = phd.mc['k'])
    #ax4.errorbar(n_trees, classifier.treedata[:, 4], yerr =classifier.treedata[:, 5] / np.sqrt(5), label ='Cross validation LDA 3d', color = phd.mc['grey'])
    #ax4.errorbar(n_trees, classifier.treedata[:, 7], yerr =classifier.treedata[:, 8] / np.sqrt(5), label ='Cross validation PCA 3d', color = phd.mc['b'])
    lw = 1.0
    ax1.plot(n_trees, classifier.treedata[:, 1], color=phd.mc['k'],linestyle='-',linewidth = lw,  label='CV 23d')
    ax1.plot(n_trees, classifier.treedata[:, 3], color=phd.mc['k'],linestyle='--', linewidth = lw, label='Test 23d')

    ax1.plot(n_trees, classifier.treedata[:, 4], color=phd.mc['r'],linestyle='-', linewidth = lw,  label='CV LDA 3d')
    ax1.plot(n_trees, classifier.treedata[:, 6], color = phd.mc['r'],linestyle = '--',linewidth = lw,  label ='Test LDA 3d')
    print classifier.treedata[:,6]

    ax1.plot(n_trees, classifier.treedata[:, 7], color = phd.mc['b'],linestyle = '-', linewidth = lw, label ='CV PCA 3d')
    ax1.plot(n_trees, classifier.treedata[:, 9], color = phd.mc['b'],linestyle = '--', linewidth = lw, label ='Test PCA 3d')

    xlim = ax1.get_xlim()
    myxlim = (-0.1, xlim[1])
    ax1.set_xlim(myxlim)

    ax1.hlines(0.25, 0, xlim[1], label = 'Chance', linewidth = lw, color = phd.mc['grey'])

    leg = ax1.legend(frameon = False,ncol = 3, loc ='best', fontsize = 10)

    ylim = ax1.get_ylim()
    myylim = (ylim[0],1)
    ax1.set_ylim(myylim)
    ax1.set_ylim(0,1)

    ax1.set_xlabel('Number of Trees')
    ax1.set_ylabel('Classifier performance')
    ax1.set_title('Random forest performance')

    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    # Disable ticks.
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.savefig('/Users/jonathan/PhD/Presentations/performance.png')
    plt.show()

def feature_importance():
    feature_labels = ['min','max','mean','skew','std','kurtosis','coastline','bl n',
                           'bl diff','bl diff skew','n pks','n vals','av pk','av val','av pk val range',
                               '1 hz','5 hz','10 hz','15 hz','20 hz','30 hz','60 hz','90 hz']

    pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
    classifier = pickle.load(open('../saved_clf','rb'))
    fig = plt.figure(figsize=(5.5,6))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Feature importance (Gini index reduction)')
    ax1.set_xlabel('Importance (%)')
    import pandas as pd
    df = pd.DataFrame(classifier.r_forest.feature_importances_*100,feature_labels)
    df = df.sort(columns=0)
    df.plot(kind='barh', ax = ax1, rot = 0, legend = False, grid = False,
            color=phd.mc['k'], )

    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
                # Disable ticks.
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('/Users/jonathan/PhD/Presentations/importance.png')
    plt.show()


if __name__ == '__main__':
    #main()
    #lda_plot()
    #pca_plot()
    #scatter_plot_matrix()
    #perfomance()
    feature_importance()

