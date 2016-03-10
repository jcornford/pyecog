'''

Simple file, focusing on loading of hdf5 stored data. Like network classifier 2.

 - double check the data
 - train the classifier
'''
import os
import pickle

import h5py
import numpy as np

import pythoncode.utils
from pythoncode.make_pdfs import plot_traces, plot_traces_hdf5
from pythoncode.extrator import FeatureExtractor
from pythoncode.classifier import NetworkClassifer

def normalise(series):
    #return series
    a = np.min(series, axis=1)
    b = np.max(series, axis=1)
    return np.divide((series - a[:, None]), (b-a)[:,None])

def make_pdfs_for_labelling_converted_ndf(converted_ndf, pdf_dir):

    with h5py.File(converted_ndf , 'r') as hf:
                for ndfkey in hf.keys():
                    print ndfkey, 'is hf key'
                    datadict = hf.get(ndfkey)

                for tid in datadict.keys():
                    data = np.array(datadict[tid]['data'])

                print data.shape

                index = data.shape[0]/ (5120/2)
                print index, 'is divded by 5120'

                data_array = np.reshape(data[:(5120/2)*index], (index,(5120/2),))
                print data_array.shape

    plot_traces_hdf5(data_array,
                    savestring = os.path.join(pdf_dir, converted_ndf.split('/')[-1]+'_'),
                    prob_thresholds= None)



def append_to_training_hdf5():
    print "need to implement!"

def make_training_hdf5():

    basedir = '/Volumes/LaCie/Albert_ndfs/training_data/raw_hdf5s/'
    filepairs = [('state_labels_2016_01_21_19-16.csv','2016_01_21_19:16.hdf5'),
                 ('state_labels_2016_01_21_13-16.csv','2016_01_21_13:16.hdf5'),
                 ('state_labels_2016_01_21_11-16.csv','2016_01_21_11:16.hdf5'),
                 ('state_labels_2016_01_21_10-16.csv','2016_01_21_10:16.hdf5'),
                 ('state_labels_2016_01_21_08-16.csv','2016_01_21_08:16.hdf5')]

    data_array_list = []
    label_list = []

    for pair in filepairs:
        labels = np.loadtxt(os.path.join(basedir, pair[0]),delimiter=',')[:,1]
        print labels.shape

        converted_ndf = os.path.join(basedir, pair[1])

        with h5py.File(converted_ndf , 'r') as hf:

                for key in hf.attrs.keys():
                    print key, hf.attrs[key]
                print hf.items()

                for ndfkey in hf.keys():
                    print ndfkey, 'is hf key'
                    datadict = hf.get(ndfkey)

                for tid in datadict.keys():

                    time = np.array(datadict[tid]['time'])
                    data = np.array(datadict[tid]['data'])
                    #print npdata.shape

                print data.shape

                index = data.shape[0]/ (5120/2)
                print index, 'is divded by 5120'

                data_array = np.reshape(data[:(5120/2)*index], (index,(5120/2),))
                print data_array.shape
                #plt.figure(figsize = (20,10))
                #plt.plot(data_array[40,:])
                data_array_list.append(data_array)
                label_list.append(labels)
                #plt.show()
    data_array = np.vstack(data_array_list)
    print data_array.shape, 'is shape of data'
    labels = np.hstack(label_list)
    print labels.shape, 'is shape of labels'

    ### Write it up! ###
    file_name = '/Volumes/LaCie/Albert_ndfs/training_data/training_data_v2.hdf5'
    with h5py.File(file_name, 'w') as f:
        f.name

        training = f.create_group('training')

        training.create_dataset('data', data = data_array)
        training.create_dataset('labels', data = labels)

        print training.keys()


def check_training_data(plot = True, correct = False):
    '''

    This function will load and then plot the data on the pdfs.

    '''
    # first load up the training data
    filename = '/Volumes/LaCie/Albert_ndfs/training_data/training_data_v2.hdf5'

    with h5py.File(filename, 'r') as f:
        print f.keys()
        train_data_raw = np.array(f['training/data'])
        train_data = pythoncode.utils.filterArray(train_data_raw, window_size= 7, order=3)
        train_data = normalise(train_data)
        train_labels= np.array(f['training/labels']).astype(int)

        print train_data.shape
        print train_labels.shape


    if plot:
        save_dir = '/Volumes/LaCie/Albert_ndfs/training_data/pdfs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_traces_hdf5(train_data,
                    train_labels,
                    savestring = os.path.join(save_dir , 'training_v2_viz'),
                    prob_thresholds= None)

    if correct:
        # correct labels here
        train_labels[126] = 4

        # saveup into new hdf5 file
        #filename = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5'
        with h5py.File(filename, 'w') as f:
            training = f.create_group('training')
            training.create_dataset('data', data = train_data_raw)
            training.create_dataset('labels', data = train_labels)

def train_classifier():

    '''

    Use the hdf5 from the check data to train and assess the output of the classifier!

    '''
    newfile = '/Volumes/LaCie/Albert_ndfs/training_data/training_data_v2.hdf5'

    with h5py.File(newfile, 'r') as f:
        print f.keys()
        test_data_raw = np.array(f['training/data'])
        train_data_raw = np.array(f['training/data'])

        test_data = pythoncode.utils.filterArray(test_data_raw, window_size= 7, order=3)
        train_data = pythoncode.utils.filterArray(train_data_raw, window_size= 7, order=3)

        test_data = normalise(test_data)
        train_data = normalise(train_data)

        test_labels = np.array(f['training/labels']).astype(int)
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
        f = open('/Volumes/LaCie/Albert_ndfs/pickled_classifier_t5dbs','wb')
        pickle.dump(classifier,f)


    #plot_traces(train_data, train_labels,savestring = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/'+'norm_filtered_check',)

def main():
    #check_training_data(plot = True)
    train_classifier()
    #make_training_hdf5()
    #make_pdfs_for_labelling_converted_ndf('/Volumes/LaCie/Gabriele/hdf5s/M1456848029.hdf5','/Volumes/LaCie/Gabriele/pdfs' )



if __name__ == "__main__":
    main()