"""
This file is now out of date. Use network_classifier_v2.

This was last run to save the training data into a hdf5 file.

"""

import pickle

import h5py
import numpy as np

import utils
from pyecog.light_code.extractor import FeatureExtractor
from pyecog.light_code.classifier import NetworkClassifer


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


################## Training data added 2016/02/15 after first try outs ################
lacie_training = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/labelling_for_training/'
training_traces2 = pickle.load(open(lacie_training + 'new_training_data_2016_02_09','rb'))
training_traces2_norm = utils.normalise(training_traces2)
training_data2 = FeatureExtractor(training_traces2_norm).feature_array
training_labels2 = pickle.load(open(lacie_training + 'new_training_labels_2016_02_09','rb'))

# now merge:

updated_training_data_traces = np.vstack((training_traces[training_indexes,:],training_traces2))
updated_training_data = np.vstack((training_data.feature_array[training_indexes,:], training_data2))
updated_training_labels = np.hstack((training_labels[training_indexes],training_labels2))
print updated_training_data.shape, 'is updated shape, and labels...',
print updated_training_labels.shape
print updated_training_data_traces.shape, 'is updated traces shape!'



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

#np.savetxt('all_traces.csv',np.vstack((validation_traces_norm,training_traces_norm)),delimiter=',')
#np.savetxt('all_cleanup.csv',np.hstack((validation_labels)), delimiter = ',')

# comment out on 2016/02/15 when using the extra training data!
#classifier = NetworkClassifer(training_data.feature_array[training_indexes,:],training_labels[training_indexes],
#                             validation_data.feature_array[validation_indexes],validation_labels[validation_indexes])

###### Save the training and test here into hdf5 / databases #######
hdf5_data = {}
#file_name = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223.hdf5'

print file_name

with h5py.File(file_name, 'w') as f:
    f.name

    training = f.create_group('training')
    test = f.create_group('test')

    training.create_dataset('data', data = updated_training_data_traces)
    test.create_dataset('data', data = validation_traces[validation_indexes])

    training.create_dataset('labels', data = updated_training_labels)
    test.create_dataset('labels', data = validation_labels[validation_indexes])

    training.create_dataset('features', data = updated_training_data)
    test.create_dataset('features',data = validation_data.feature_array[validation_indexes])

    print training.keys()
    print test.keys()

    #plt.plot(training['data'][49,:])

#exit()

classifier = NetworkClassifer(updated_training_data,updated_training_labels,
                              validation_data.feature_array[validation_indexes],validation_labels[validation_indexes])
classifier.run()

classifier.pca(n_components = 3)
classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
classifier.lda_run()
classifier.pca_run()
#classifier.randomforest_info(max_trees=2000, step = 50)
f = open('../pickled_classifier','wb')
pickle.dump(classifier,f)

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