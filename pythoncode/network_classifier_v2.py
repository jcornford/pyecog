'''

Simple file, focusing on loading of hdf5 stored data.

 - double check the data
 - train the classifier
'''

import h5py
import pickle

import numpy as np
import matplotlib.pyplot as plt

import utils
from make_pdfs import plot_traces
from extrator import FeatureExtractor
from classifier import NetworkClassifer

def normalise(series):
    #return series
    a = np.min(series, axis=1)
    b = np.max(series, axis=1)
    return np.divide((series - a[:, None]), (b-a)[:,None])


def check_data():
    '''

    This function will load and then plot the data on the pdfs.

    '''

    filename = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223.hdf5'

    with h5py.File(filename, 'r') as f:
        print f.keys()
        test_data_raw = np.array(f['test/data'])
        train_data_raw = np.array(f['training/data'])

        test_data = utils.filterArray(test_data_raw,window_size= 21, order=3)
        train_data = utils.filterArray(train_data_raw,window_size= 15, order=3)

        test_data = normalise(test_data)
        train_data = normalise(train_data)

        test_labels = np.array(f['test/labels']).astype(int)
        train_labels= np.array(f['training/labels']).astype(int)

        # as a result of eyeballing....
        train_labels[605] = 2
        train_labels[494] = 2
        train_labels[26] = 2
        train_labels[95] = 2
        train_labels[92] = 2
        train_labels[280] = 1
        train_labels[96] = 2
        train_labels[482] = 2
        train_labels[485] = 2
        train_labels[476] = 2
        train_labels[477] = 1
        train_labels[483] = 2
        train_labels[519] = 4
        train_labels[520] = 3
        train_labels[525] = 1
        train_labels[536] = 3
        train_labels[528] = 2
        train_labels[529] = 2
        train_labels[526] = 1
        train_labels[558] = 2
        train_labels[559] = 1
        train_labels[560] = 2
        train_labels[561] = 3
        train_labels[567] = 2
        train_labels[568] = 3
        train_labels[571] = 3
        train_labels[574] = 2
        train_labels[576] = 2
        train_labels[578] = 2
        train_labels[575] = 3
        train_labels[577] = 3
        train_labels[580] = 3
        train_labels[613] = 2
        train_labels[626] = 4
        train_labels[701] = 2
        train_labels[700] = 3
        train_labels[702] = 3
        train_labels[698] = 1
        train_labels[681] = 2
        train_labels[717] = 3
        train_labels[731] = 1
        train_labels[570] = 2

        test_labels[37] = 3
        print train_data.shape
        print train_labels.shape

    plot_traces(test_data,
                test_labels,
                savestring = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/'+'norm_filtered_test',
                prob_thresholds= None)

    # saveup into new hdf5 file
    newfile = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5'
    with h5py.File(newfile, 'w') as f:
        f.name

        training = f.create_group('training')
        test = f.create_group('test')

        training.create_dataset('data', data = train_data_raw)
        test.create_dataset('data', data = test_data_raw)

        training.create_dataset('labels', data = train_labels)
        test.create_dataset('labels', data = test_labels)


def train_classifier():

    '''

    Use the hdf5 from the check data to train and assess the output of the classifier!

    '''
    newfile = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5'

    with h5py.File(newfile, 'r') as f:
        print f.keys()
        test_data_raw = np.array(f['test/data'])
        train_data_raw = np.array(f['training/data'])

        test_data = utils.filterArray(test_data_raw,window_size= 7, order=3)
        train_data = utils.filterArray(train_data_raw,window_size= 7, order=3)

        test_data = normalise(test_data)
        train_data = normalise(train_data)


        test_labels = np.array(f['test/labels']).astype(int)
        train_labels= np.array(f['training/labels']).astype(int)


        training_features = FeatureExtractor(train_data).feature_array#[:,[2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22]]
        test_features = FeatureExtractor(test_data).feature_array#[:,[2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22]]

        classifier = NetworkClassifer(training_features,train_labels, test_features, test_labels)
        classifier.run()

        classifier.pca(n_components = 3)
        classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
        classifier.lda_run()
        classifier.pca_run()


        #f = open('../pickled_classifier_20160223','wb')
        f = open('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/pickled_classifier_20160223','wb')
        pickle.dump(classifier,f)


    #plot_traces(train_data, train_labels,savestring = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/'+'norm_filtered_check',)

def main():
    #check_data()
    train_classifier()

if __name__ == "__main__":
    main()