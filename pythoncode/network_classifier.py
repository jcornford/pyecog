import time

import matplotlib.pyplot as plt
import numpy as np

#import stfio_plot as sp
from load_seizure_data import SeizureData
from relabeling_functions import relabel,reorder
from nc import extract_features
from classifier import NetworkClassifer

def normalise(series):
    a = np.min(series, axis=1)
    b = np.max(series, axis=1)
    return np.divide((series - a[:, None]), (b-a)[:,None])

FROMSCRATCH = False
if FROMSCRATCH:
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


    print time.clock()
    features0616 = extract_features(data0616)
    print time.clock()

    np.savetxt('../features0616_n342.csv',features0616,delimiter=',')
    np.savetxt('../labels0616_n342.csv',labels0616,delimiter=',')

else:
    labels0616 = np.loadtxt('../labels0616_n342.csv', delimiter=',')
    features0616 = np.loadtxt('../features0616_n342.csv', delimiter=',')

    f129 = np.loadtxt('../features_orginal301_n129.csv', delimiter=',')
    l129 = np.loadtxt('../labels_orginal301_n129.csv', delimiter=',')

    f342 = np.loadtxt('../features0616_n342.csv', delimiter=',')
    l342 = np.loadtxt('../labels0616_n342.csv', delimiter=',')

    f471 = np.vstack([f129,f342])
    l471 = np.hstack([l129,l342])

    validation_features = np.loadtxt('../val_feats_fair.csv', delimiter=',')
    validation_labels = np.loadtxt('../val_labels_fair.csv', delimiter=',')

print features0616.shape
#classifier0616 = NetworkClassifer(features0616,labels0616, validation_features,validation_labels)
classifier0616 = NetworkClassifer(f471,l471, validation_features,validation_labels)
classifier0616.run()
classifier0616.randomforest_info()