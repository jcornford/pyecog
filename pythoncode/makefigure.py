import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

treedata = np.loadtxt('../treedata.csv',delimiter = ',')

print treedata.shape

figure = plt.figure()
grid =  gridspec.GridSpec(2, 1,)

ax1 = plt.subplot(grid[0])
ax1.plot(treedata[:,0], label = 'Training')
ax1.plot(treedata[:,1], label = 'Test')
ax1.plot(treedata[:,2], label = 'Validation')
#ax1.hlines(0.25,0,treedata.shape[0], linestyle = '--')
ax1.legend(frameon = False, loc = 'best', fontsize = 10)
#ax1.set_ylim(0,1)

#ax2 = plt.subplot(grid[1])

plt.show()