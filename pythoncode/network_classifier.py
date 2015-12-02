import time

import pickle
import matplotlib.pyplot as plt
import numpy as np

#import stfio_plot as sp
from network_loader import SeizureData
from relabeling_functions import relabel,reorder
from extrator import FeatureExtractor
from classifier import NetworkClassifer

#from make_pdfs import plot_traces

print 'N.B. move normalisation to loader'
def normalise(series):
    a = np.min(series, axis=1)
    b = np.max(series, axis=1)
    return np.divide((series - a[:, None]), (b-a)[:,None])

FROMSCRATCH = True
if FROMSCRATCH:
    ################# 'NEW data' ###################
    dirpath = '/Users/Jonathan/PhD/Seizure_related/20150616'
    _20150616dataobj = SeizureData(dirpath, amount_to_downsample = 40)
    _20150616dataobj.load_data()
    _20150616data = _20150616dataobj.data_array
    _20150616labels = _20150616dataobj.label_colarray
    _20150616data_norm = normalise(_20150616data)

    print 'Done'

    print _20150616dataobj.filename_list.shape

    _20150616dataobj.filenames_list = [_20150616dataobj.filename_list[i] for i in range(_20150616dataobj.filename_list.shape[0])]

    for name in _20150616dataobj.filenames_list[0:20]:
        print name[-34:]

    # select out the stuff we want
    #inds = np.loadtxt('0901_400newdata.csv', delimiter=',')
    notebook_dir = '/Users/jonathan/PhD/Seizure_related/2015_08_PyRanalysis/'
    inds = np.loadtxt(notebook_dir +'0616correctedintervals.csv', delimiter=',')

    print len(inds)
    data0616_unnorm = _20150616data[list(inds[:,0])]
    data0616 = _20150616data_norm[list(inds[:,0])]
    labels0616 = _20150616labels[list(inds[:,0])]
    print data0616_unnorm.shape
    for i in range(data0616.shape[0]):
        labels0616[i] = inds[i,1]

    print len(labels0616)

    ################## Original Data ####################
    dirpath = '/Users/Jonathan/PhD/Seizure_related/Classified'
    dataobj = SeizureData(dirpath,amount_to_downsample = 20)
    dataobj.load_data()
    dataobj = relabel(dataobj)
    dataobj = reorder(dataobj)
    dataset301 = dataobj.data_array
    labels301 = dataobj.label_colarray
    print dataset301.shape
    print labels301.shape
    new_labels = np.loadtxt(notebook_dir+'new_event_labels_28082015.csv',delimiter= ',')
    for x in new_labels:
        labels301[x[0]] = x[1]

    selection = np.loadtxt(notebook_dir+'perfect_event_labels_28082015.csv',delimiter= ',')
    indexes =  list(selection[:,0])
    print indexes

    dataset129_unnorm = dataset301[indexes,:]
    dataset129_norm = normalise(dataset129_unnorm)
    dataset301_norm = normalise(dataset301)
    labels129 = labels301[indexes]

    print dataset129_norm.shape

    ################## Validation Data ####################


    #f = open('../dataset','wb')
    #pickle.dump(data0616,f)
    #print 'Loaded and pickled'

    #features0616 = extract_features(data0616)

    #np.savetxt('../features0616_n342.csv',features0616,delimiter=',')
    #np.savetxt('../labels0616_n342.csv',labels0616,delimiter=',')



else:
    labels0616 = np.loadtxt('../labels0616_n342.csv', delimiter=',')
    features0616 = np.loadtxt('../features0616_n342.csv', delimiter=',')

    f129 = np.loadtxt('../features_orginal301_n129.csv', delimiter=',')
    l129 = np.loadtxt('../labels_orginal301_n129.csv', delimiter=',')

    f342 = np.loadtxt('../features0616_n342.csv', delimiter=',')
    l342 = np.loadtxt('../labels0616_n342.csv', delimiter=',')

    f471 = np.vstack([f129,f342])
    l471 = np.hstack([l129,l342])

labels0616 = np.loadtxt('../labels0616_n342.csv', delimiter=',')
features0616 = np.loadtxt('../features0616_n342.csv', delimiter=',')
validation_features = np.loadtxt('../val_feats_fair.csv', delimiter=',')
validation_labels = np.loadtxt('../val_labels_fair.csv', delimiter=',')

print data0616.shape
print data0616.shape

featuredata1 = FeatureExtractor(data0616)
featuredata2 = FeatureExtractor(dataset129_norm)

labels129_flat = np.ravel(labels129)
labels0616_flat = np.ravel(labels0616)

print labels129_flat.shape, labels0616.shape
labels_full = np.hstack([labels0616_flat,labels129_flat])
featuredata_full = np.vstack([featuredata1.feature_array,featuredata2.feature_array])
#classifier0616 = NetworkClassifer(features,labels, validation_features,validation_labels)
classifier = NetworkClassifer(featuredata_full,labels_full, validation_features,validation_labels)
#classifier = NetworkClassifer(features0616,labels0616, validation_features,validation_labels)
classifier.run()
#classifier.randomforest_info()
classifier.pca(n_components = 2)
classifier.lda(n_components = 2, pca_reg = True)
f = open('../saved_clf','wb')
pickle.dump(classifier,f)

