import time

import pickle
import matplotlib.pyplot as plt
import numpy as np

#import stfio_plot as sp
from network_loader import SeizureData
from relabeling_functions import relabel,reorder
from extrator import FeatureExtractor
from classifier import NetworkClassifer

from make_pdfs import plot_traces

print 'N.B. move normalisation to loader'
def normalise(series):
    #return series
    a = np.min(series, axis=1)
    b = np.max(series, axis=1)
    return np.divide((series - a[:, None]), (b-a)[:,None])

reload_training = False
if reload_training:
    ################# 'NEW data' ###################
    dirpath = '/Users/Jonathan/PhD/Seizure_related/20150616'
    _20150616dataobj = SeizureData(dirpath, amount_to_downsample = 40)
    _20150616dataobj.load_data()
    _20150616data = _20150616dataobj.data_array
    _20150616labels = _20150616dataobj.label_colarray
    _20150616data_norm = normalise(_20150616data)

    print _20150616dataobj.filename_list.shape
    _20150616dataobj.filenames_list = [_20150616dataobj.filename_list[i] for i in range(_20150616dataobj.filename_list.shape[0])]
    for name in _20150616dataobj.filenames_list[0:20]:
        print name[-34:]

    # select out the stuff we want
    #inds = np.loadtxt('0901_400newdata.csv', delimiter=',')
    notebook_dir = '/Users/jonathan/PhD/Seizure_related/2015_08_PyRanalysis/'
    inds = np.loadtxt(notebook_dir +'0616correctedintervals.csv', delimiter=',')
    data0616_unnorm = _20150616data[list(inds[:,0])]
    data0616 = _20150616data_norm[list(inds[:,0])]
    labels0616 = _20150616labels[list(inds[:,0])]
    for i in range(data0616.shape[0]):
        labels0616[i] = inds[i,1]

    ################## Original Data ####################
    dirpath = '/Users/Jonathan/PhD/Seizure_related/Classified'
    dataobj = SeizureData(dirpath,amount_to_downsample = 20)
    dataobj.load_data()
    dataobj = relabel(dataobj)
    dataobj = reorder(dataobj)
    dataset301 = dataobj.data_array
    labels301 = dataobj.label_colarray
    new_labels = np.loadtxt(notebook_dir+'new_event_labels_28082015.csv',delimiter= ',')
    for x in new_labels:
        labels301[x[0]] = x[1]

    selection = np.loadtxt(notebook_dir+'perfect_event_labels_28082015.csv',delimiter= ',')
    indexes =  list(selection[:,0])
    dataset129_unnorm = dataset301[indexes,:]
    dataset129_norm = normalise(dataset129_unnorm)
    dataset301_norm = normalise(dataset301)
    labels129 = labels301[indexes]

elif not reload_training:
    print 'skipping reload training'
    training_traces = pickle.load(open('../full_raw_training','rb'))
    training_traces_norm = normalise(training_traces)
    training_data = FeatureExtractor(training_traces_norm)

################# Training cleanup ###################
cleanup = np.loadtxt('../Training_cleanup.csv',delimiter=',')
training_labels = np.array([int(x[1]) for x in cleanup])

training_indexes = []
for i in range(training_labels.shape[0]):
        if training_labels[i] != 0:
            training_indexes.append(i)

################## Validation Data ####################
dirpath = '/Users/Jonathan/PhD/Seizure_related/batchSept_UC_20'
testdataobj20 = SeizureData(dirpath,amount_to_downsample = 40)
testdataobj20.load_data()
datasettest20 = testdataobj20.data_array

dirpath = '/Users/Jonathan/PhD/Seizure_related/batchSept_UC_40'
testdataobj40 = SeizureData(dirpath,amount_to_downsample = 40)
testdataobj40.load_data()
datasettest40 = testdataobj40.data_array

datasettest = np.vstack([datasettest20,datasettest40])
datasettest_norm = normalise(datasettest)
validation_data = FeatureExtractor(datasettest_norm)
datasettest_norm.shape
notebook_dir = '/Users/jonathan/PhD/Seizure_related/2015_08_PyRanalysis/classifier_writeup/'

################# Validation cleanup ###################
cleanup = np.loadtxt('../validation_cleanUp.csv',delimiter=',')
validation_labels = np.array([int(x[1]) for x in cleanup])

validation_indexes = []
for i in range(validation_labels.shape[0]):
        if validation_labels[i] != 0:
            validation_indexes.append(i)


#val_labels = np.loadtxt(notebook_dir+'sept_279_labels.csv', delimiter=',')
#val_uncertain = np.loadtxt(notebook_dir+'sept_279_uncertain_events.csv',delimiter=',')
#val_uncertain = np.array([int(x) for x in val_uncertain])
#val_uncertain = []
#ok_indexes = []
#for i in range(validation_data.feature_array.shape[0]):
#    if i not in val_uncertain:
#        ok_indexes.append(i)
#validation_data.feature_array_fair = validation_data.feature_array[ok_indexes,:]
#validation_labels_fair = val_labels[ok_indexes]

''' ######## for pdfs of the validation ############ '''
#f = open('../validation_label_traces_tuple','wb')
#pickle.dump((validation_labels ,datasettest_norm[ok_indexes,:]),f)


######### Feature selection ###########
feature_labels = ['0min','1max','2mean','3skew','4std','5kurtosis','6sum of absolute difference','7baseline_n',
                               '8baseline_diff','9baseline_diff_skew','10n_pks','11n_vals','12av_pk','13av_val','14av pk val range',
                               '151 hz','165 hz','1710 hz','1815 hz','1920 hz','20_30 hz','21_60 hz','22_90 hz']
fis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
fis = [0,1,2,3,4,5,6,7,8,9,12,14,15,16,17,18,19]
#classifier0616 = NetworkClassifer(features,labels, validation_features,validation_labels)
classifier = NetworkClassifer(training_data.feature_array[training_indexes,:],training_labels[training_indexes],
                              validation_data.feature_array[validation_indexes],validation_labels[validation_indexes])
#classifier = NetworkClassifer(features0616,labels0616, validation_features,validation_labels)
classifier.run()

classifier.randomforest_info(max_trees=2000)

classifier.pca(n_components = 3)
classifier.lda(n_components = 3, pca_reg = True, reg_dimensions = 10)
f = open('../saved_clf','wb')
pickle.dump(classifier,f)


'''######## for pdfs of the original features ############'''
#training_traces = np.vstack((data0616,dataset301_norm))
#training_traces = training_data.feature_array[ok_indexes,:]
#f = open('../training_label_traces_tuple','wb')
#pickle.dump((training_indexes,training_traces),f)

#f = open('../full_raw_training','wb')
#pickle.dump(np.vstack([data0616_unnorm,dataset301]),f)

'''
testname_list = [testdataobj.filename_list[i] for i in range(testdataobj.filename_list.shape[0])]
len(testname_list)

info = []
for i, name in enumerate(testname_list):
    #print i,name[-34:]," State:" +str(pred_labels_new[i]), str(np.max(pred_labels_new_pr[i,:])*100)+'%'
    info.append([i,name[-40:]," State:" +str(pred_labels_new[i]), str(np.max(pred_labels_new_pr[i,:])*100)+'%'])
info = np.array(info)
print info.shape

'''