import matplotlib.pyplot as plt
import numpy as np


from loadSeizureData  import LoadSeizureData
from classifierTester import ClassifierTester
from basicFeatures    import BasicFeatures
from randomForestClassifier import RandomForest

dirpath = '/Users/Jonathan/Documents/PhD /Seizure_related/Network_states/VMData/Classified'
dataobj = LoadSeizureData(dirpath)
dataobj.load_data()

colors = ['k','g','b','r']
labelslist = dataobj.label_colarray[:,0].tolist()
newlabellist = []
plt.figure(figsize = (12,6))
plt.show()

for i in [1,2]:#range(dataobj.data_array.shape[0]):
    #print colors[labelslist[i]]
    plt.plot(dataobj.data_array[i,:],colors[labelslist[i]], alpha = 1 )
    
    print 'This was orignally state ', labelslist[i], ' !'
    newlabellist.append(raw_input('Please enter correct network state'))
    #plt.clf()
    
print newlabellist
np.save('newstates',newlabellist )

