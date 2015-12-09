
from network_loader import SeizureData
from extrator import FeatureExtractor

import pickle
import numpy as np
class Predictor():

    def __init__(self, clf_pickle_path = None):
        if clf_pickle_path == None:
            clf_pickle_path = '../saved_clf'
        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.r_forest = self.classifier.r_forest
        print self.r_forest

pr = Predictor()


#cleanup = np.loadtxt('../Training_cleanup.csv',delimiter=',')
#labels = np.array([int(x[1]) for x in cleanup])
#print labels
#ok_indexes = []
#for i in range(labels.shape[0]):
#        if labels[i] != 0:
#            ok_indexes.append(i)
#print labels[ok_indexes]
    #validation_data.feature_array_fair = validation_data.feature_array[ok_indexes,:]
    #validation_labels_fair = val_labels[ok_indexes]

#new_labels = np.loadtxt(notebook_dir+'new_event_labels_28082015.csv',delimiter= ',')
 #   for x in new_labels:
  #      labels301[x[0]] = x[1]

   # selection = np.loadtxt(notebook_dir+'perfect_event_labels_28082015.csv',delimiter= ',')
    #indexes =  list(selection[:,0])
    #print indexes