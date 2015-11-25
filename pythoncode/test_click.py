# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 22:18:15 2014
Not using this!
@author: Jonathan
"""

from matplotlib import pyplot as plt

from load_seizure_data  import LoadSeizureData
from classifier_tester import ClassifierTester
from basicFeatures    import BasicFeatures
from randomForestClassifier import RandomForest


COUNT = 0
def plot_next():
	print 'id plot somethine else now'
	global COUNT
	plt.cla()
	plt.plot(dataobj.data_array[COUNT,:])
	COUNT +=1
	print COUNT 

def change_label():
	print 'change label'

def onclick(event):
    button=event.button
    x=event.xdata
    y=event.ydata

    if button==1: 
    	plot_next()
    if button!=1: 
    	change_label()
 
def loadup():
	dirpath = '/Users/Jonathan/Documents/PhD /Seizure_related/Network_states/VMData/Classified'
	dataobj = LoadSeizureData(dirpath)
	dataobj.load_data()

	return dataobj

fig = plt.figure()
dataobj = loadup()
plot_next()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()