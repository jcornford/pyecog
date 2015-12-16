import time

import pickle
import matplotlib.pyplot as plt
import numpy as np

import utils
from network_loader import SeizureData
from relabeling_functions import relabel,reorder
from extrator import FeatureExtractor
from classifier import NetworkClassifer
from make_pdfs import plot_traces

################# Training Data ###################
reload_training = False
if reload_training:
    training_traces = utils.raw_training_load()
    training_traces_norm = utils.normalise(training_traces)
    training_data = FeatureExtractor(training_traces_norm)
    #f = open('../full_raw_training','wb')
    #pickle.dump(training_traces,f)

elif not reload_training:
    print 'skipping raw training load'
    training_traces = pickle.load(open('../full_raw_training','rb'))
    training_traces_norm = utils.normalise(training_traces)
    training_data = FeatureExtractor(training_traces_norm)
    np.savetxt('training_traces.csv',training_traces_norm,delimiter=',')

################# Training Labels and mixed event exclusion ###################
cleanup = np.loadtxt('../Training_cleanup.csv',delimiter=',')
training_labels = np.array([int(x[1]) for x in cleanup])
print training_labels.shape
training_indexes = []
for i in range(training_labels.shape[0]):
        if training_labels[i] != 0:
            training_indexes.append(i)

################## Test Data ####################
reload_validation = False
if reload_validation:
    validation_traces = utils.raw_validation_load()
    validation_traces_norm = utils.normalise(validation_traces)
    validation_data = FeatureExtractor(validation_traces_norm)
    #f = open('../raw_validation','wb')
    #pickle.dump(validation_traces,f)

elif not reload_training:
    print 'skipping raw validation load'
    validation_traces = pickle.load(open('../raw_validation','rb'))
    validation_traces_norm = utils.normalise(validation_traces)
    validation_data = FeatureExtractor(validation_traces_norm)
    np.savetxt('test_traces.csv',validation_traces_norm,delimiter=',')

################# Validation cleanup ###################
cleanup = np.loadtxt('../validation_cleanUp.csv',delimiter=',')
validation_labels = np.array([int(x[1]) for x in cleanup])

validation_indexes = []
for i in range(validation_labels.shape[0]):
        if validation_labels[i] != 0:
            validation_indexes.append(i)

np.savetxt('all_traces.csv',np.vstack((validation_traces_norm,training_traces_norm)),delimiter=',')
np.savetxt('all_cleanup.csv',np.hstack((validation_labels)), delimiter = ',')
classifier = NetworkClassifer(training_data.feature_array[training_indexes,:],training_labels[training_indexes],
                              validation_data.feature_array[validation_indexes],validation_labels[validation_indexes])
classifier.run()

classifier.pca(n_components = 3)
classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
classifier.lda_run()
classifier.pca_run()
#classifier.randomforest_info(max_trees=2000, step = 50)
#f = open('../saved_clf','wb')
#pickle.dump(classifier,f)

''' ######## for pdfs of the validation ############ '''
#f = open('../validation_label_traces_tuple','wb')
#pickle.dump((validation_labels ,datasettest_norm[ok_indexes,:]),f)

'''######## for pdfs of the original features ############'''
#training_traces = np.vstack((data0616,dataset301_norm))
#training_traces = training_data.feature_array[ok_indexes,:]
#f = open('../training_label_traces_tuple','wb')
#pickle.dump((training_indexes,training_traces),f)

'''
testname_list = [testdataobj.filename_list[i] for i in range(testdataobj.filename_list.shape[0])]
len(testname_list)

info = []
for i, name in enumerate(testname_list):
    #print i,name[-34:]," State:" +str(pred_labels_new[i]), str(np.max(pred_labels_new_pr[i,:])*100)+'%'
    info.append([i,name[-40:]," State:" +str(pred_labels_new[i]), str(np.max(pred_labels_new_pr[i,:])*100)+'%'])
info = np.array(info)
print info.shape

######### Feature selection ###########
feature_labels = ['0min','1max','2mean','3skew','4std','5kurtosis','6sum of absolute difference','7baseline_n',
                               '8baseline_diff','9baseline_diff_skew','10n_pks','11n_vals','12av_pk','13av_val','14av pk val range',
                               '151 hz','165 hz','1710 hz','1815 hz','1920 hz','20_30 hz','21_60 hz','22_90 hz']
fis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
fis = [0,1,2,3,4,5,6,7,8,9,12,14,15,16,17,18,19]

'''