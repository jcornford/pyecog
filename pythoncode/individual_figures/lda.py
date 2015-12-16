import pickle

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pythoncode import utils as phd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

pth = '/Users/jonathan/PycharmProjects/networkclassifer/saved_clf'
pickle.load(open('../../saved_clf','rb'))

'''

fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
lda = classifier.lda_iss_features
for i in range(pca.shape[0]):
    ax1.scatter(lda[i,0], lda[i,1],lda[i,2], c = phd.mc[cs[int(classifier.labels[i])]], edgecolor = 'k',
                s = 20,linewidth = 0.1, depthshade = True, alpha = 0.8)

ax1.set_xlabel("LD 1",)
ax1.set_ylabel("LD 2",)
ax1.set_zlabel("LD 3")
ax1.view_init(elev=-174., azim=73)
#ax1.set_zlim((-5,5))
#ax1.set_xlim((-4.5,6))
#ax1.set_ylim((-5,12))
ax1.set_title('3d LDA projection')


plt.show()
'''