'''
Simple file, focusing on loading and training on hdf5 stored data.


- To do:
  1. Easy appending of new data to training dictionary (work in parralel with the hdf5 stuff)
'''

import h5py
import pickle

import numpy as np

import utils
from make_pdfs import plot_traces
from extrator import FeatureExtractor
from classifier import NetworkClassifer

class ClassifierHandler():

    def __init__(self):
        print 'Started'
        self.annotated_data_path = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223.hdf5'
        self.filename = None

        self.train_data_raw = None
        self.test_data_raw  = None
        self.test_labels    = None
        self.train_labels   = None

        # This might be different from raw as normalised or filtered etc
        self.test_data  = None
        self.train_data = None

    def load_hdf5_test_train_traces(self, filename):
        self.filename = filename

        with h5py.File(filename, 'r') as f:
            print f.keys()

            self.test_data_raw = np.array(f['test/data'])
            self.train_data_raw = np.array(f['training/data'])

            test_data = utils.filterArray(self.test_data_raw, window_size= 7, order=3)
            train_data = utils.filterArray(self.train_data_raw, window_size= 7, order=3)

            self.test_data = self._normalise(test_data)
            self.train_data = self._normalise(train_data)

            self.test_labels = np.array(f['test/labels']).astype(int)
            self.train_labels= np.array(f['training/labels']).astype(int)

    def correct_labels(self, correction_tuple_list, set_string):
        '''

        Args:
            correction_tuple_list: list containing tuples (index, value)
            set_string: 'test' or 'train'

        TODO: Add handling for single tuple and checking for errors
        '''
        for tup in correction_tuple_list:
            index = tup[0]
            value = tup[1]
            if set_string == 'test':
                self.test_labels[index] = value
            elif set_string == 'train':
                self.train_labels[index] = value
            else:
                print 'Invalid set_string argument:', set_string
                exit()

    def check_data(self, pdf_savepath):
        '''
        This function will load and then plot the data on the pdfs.

        TODO: Be able to specify ranges in future.
        '''
        plot_traces(self.test_data, labels = self.test_labels, savepath = pdf_savepath+'test')
        plot_traces(self.train_data, labels = self.train_labels, savepath = pdf_savepath+'train')

    def save_new_hdf5_test_train(self, filepath, overwrite_flag = False):

        # saveup into new hdf5 file
        new_filepath = filepath
        if new_filepath == self.filename:

            if overwrite_flag == False:
                print ' New filepath will overwrite previous file, please pass overwrite_flag a value of true or' \
                      ' provide alternative filepath. Exiting...'
                exit()

            with h5py.File(new_filepath, 'w') as f:
                training = f.create_group('training')
                test = f.create_group('test')

                training.create_dataset('data', data = self.train_data_raw)
                test.create_dataset('data', data = self.test_data_raw)

                training.create_dataset('labels', data = self.train_labels)
                test.create_dataset('labels', data = self.test_labels)

    def save_feature_hdf5(self, filepath, overwrite_flag = False):
        '''
        Method saves a hdf5 file with raw_traces, labels and features:

        TODO: Save the metadata along with the file, so it is documented how the features
        were extracted from the raw traces.
        '''

        new_filepath = filepath
        if new_filepath == self.filename:

            if overwrite_flag == False:
                print ' New filepath will overwrite previous file, please pass overwrite_flag a value of true or' \
                      ' provide alternative filepath. Exiting...'
                exit()

        with h5py.File(new_filepath, 'w') as f:
                training = f.create_group('training')
                test = f.create_group('test')

                training.create_dataset('data', data = self.train_data_raw)
                test.create_dataset('data', data = self.test_data_raw)

                training.create_dataset('labels', data = self.train_labels)
                test.create_dataset('labels', data = self.test_labels)

                training.create_dataset('features', data = self.training_features)
                test.create_dataset('features', data = self.test_features)

    def extract_features(self):

        self.training_features = FeatureExtractor(self.train_data).feature_array
        self.test_features = FeatureExtractor(self.test_data).feature_array


    def train_classifier(self, training_db_path, params = {}):
        '''

        Use the hdf5 from the checked data to train and assess the output of the classifier!

        TODO:
        Assert that it has the features etc
        Pass the params dictionary to the random forest?!
        Allow user to choose own classifier
        '''

        with h5py.File(training_db_path, 'r') as f:
            train_labels= np.array(f['training/labels']).astype(int)
            test_labels = np.array(f['test/labels']).astype(int)

            training_features = np.array(f['training/features'])
            test_features = np.array(f['test/features'])

        self.classifier = NetworkClassifer(training_features, train_labels, test_features, test_labels)
        self.classifier.run()

        self.classifier.pca(n_components = 3)
        self.classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
        self.classifier.lda_run()
        self.classifier.pca_run()

    def save_classifier(self, savepath):
            f = open(savepath,'wb')
            pickle.dump(self.classifier, f)

    def load_classifier(self):
        print 'TODO'

    def assess_classifier(self):
        print 'TODO'

    def tune_classifier(self):
        '''
        Pass in dictionary of parameters over which to sweep
        Choose the type of search
        Scoring

        Returns:
            Models according to performance.

        '''
        print 'TODO'

    @ staticmethod
    def _normalise(series):
        a = np.min(series, axis=1)
        b = np.max(series, axis=1)
        return np.divide((series - a[:, None]), (b-a)[:,None])

        #plot_traces(train_data, train_labels,savestring = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/'+'norm_filtered_check',)

def main():
    handler = ClassifierHandler()
    handler.load_hdf5_test_train_traces(filename = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5')

    #handler.correct_labels([(650,3)], 'train')
    #handler.check_data(pdf_savepath = '/Volumes/LACIE SHARE/VM_data/jonny_playing/'+'norm_filtered_')
    #handler.save_new_hdf5_test_train(filepath='/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5', overwrite_flag=False)

    #handler.extract_features()
    #handler.save_feature_hdf5('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_201603_18_features.hdf5')

    handler.train_classifier('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_201603_18_features.hdf5')
    handler.save_classifier('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/pickled_classifier_20160318')

if __name__ == "__main__":
    main()