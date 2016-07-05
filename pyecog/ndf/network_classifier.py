'''
Simple file, focusing on loading and training on hdf5 stored data.


- To do:
  1. Easy appending of new data to training dictionary (work in parralel with the hdf5 stuff)
'''
from __future__ import print_function
import pickle
import os

import h5py
import numpy as np
from sklearn.cross_validation import train_test_split

import utils
from pyecog.light_code.make_pdfs import plot_traces
from pyecog.light_code.extractor import FeatureExtractor
from pyecog.light_code.classifier import NetworkClassifer


class ClassifierHandler():

    def __init__(self):
        self.annotated_data_path = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223.hdf5'
        self.filename = None

        self.train_data_raw = None
        self.test_data_raw  = None
        self.test_labels    = None
        self.train_labels   = None

        # This might be different from raw as normalised or filtered etc
        self.test_data  = None
        self.train_data = None

        self.classifier = None



    def load_labeled_traces(self, filename):
        '''

        Args:
            filename: File can either already have training and test sets, or just training.

            If just training 25% is taken as test set for future...

        Returns:

        '''
        #self.filename = os.path.abspath(filename)
        self.filename = filename

        with h5py.File(self.filename, 'r') as f:
            if len(f.keys()) == 2:
                self.test_data_raw = np.array(f['test/data'])
                self.train_data_raw = np.array(f['training/data'])
                self.test_labels = np.array(f['test/labels']).astype(int)
                self.train_labels= np.array(f['training/labels']).astype(int)
            else:
                print('Warning: Automatically assigning test and training datasets')
                self.train_data_raw, self.test_data_raw, self.train_labels, self.test_labels = train_test_split(
                    np.array(f['training/data']), np.array(f['training/labels']).astype(int), random_state= 7)

            test_data = utils.filterArray(self.test_data_raw, window_size= 7, order=3)
            train_data = utils.filterArray(self.train_data_raw, window_size= 7, order=3)

            self.test_data = self._normalise(test_data)
            self.train_data = self._normalise(train_data)

    def load_labeled_features(self, filename):
        '''

        Args:
            filename: File needs to have both test and training

        Returns:

        '''

        with h5py.File(filename, 'r') as f:
            print(f.keys())

            self.train_labels= np.array(f['training/labels']).astype(int)
            self.test_labels = np.array(f['test/labels']).astype(int)

            self.training_features = np.array(f['training/features'])
            self.test_features = np.array(f['test/features'])


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
                print('Invalid set_string argument:', set_string)
                return 0

    def check_trace_labels(self, savedir = None, range = 'all', format = '.pdf'):
        '''
        This function will load and then plot the data on the pdfs.

        Range: either 'all', or (start, end,'train' or 'test')

        TODO: Be able to specify ranges in future.
        '''
        if savedir is None:
            path =  os.path.abspath(self.filename)
            savedir = os.path.join(os.path.dirname(path), str(self.filename)+'_plots')


        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if range is 'all':
            print('Plotting test set')
            plot_traces(self.test_data, labels = self.test_labels, savepath = savedir, filename = 'test', format_string = format)
            print('Plotting training set')
            plot_traces(self.train_data, labels = self.train_labels, savepath = savedir,filename = 'train', format_string = format)

        else:
            if range[-1] == 'test':
                print('Plotting test set, from '+str(range[0])+' to '+str(range[1])+ ' will start at figure number '+str((range[0]+1)/40))
                plot_traces(self.test_data[range[0]:range[1],:], start= range[0], labels = self.test_labels[range[0]:range[1]],
                            savepath = savedir, filename = 'test', format_string = format)

            elif range[-1] == 'train':
                print('Plotting training set, from  '+str(range[0])+' to '+str(range[1])+' will start at figure number '+str((range[0]+1)/40))
                plot_traces(self.train_data[range[0]:range[1],:], start = range[0], labels = self.train_labels[range[0]:range[1]],
                            savepath = savedir, filename = 'train', format_string = format)

            else:
                print('Please specify test or training set')

    def save_labeled_traces(self, filepath, overwrite_flag = False):

        # saveup into new hdf5 file
        new_filepath = os.path.abspath(filepath)
        print('saving database at '+ new_filepath)
        if new_filepath == self.filename:

            if overwrite_flag == False:
                print(' New filepath will overwrite previous file, please pass overwrite_flag a value of true or provide alternative filepath. Exiting...')
                return 0

        with h5py.File(new_filepath, 'w') as f:
            training = f.create_group('training')
            test = f.create_group('test')

            training.create_dataset('data', data = self.train_data_raw)
            test.create_dataset('data', data = self.test_data_raw)

            training.create_dataset('labels', data = self.train_labels)
            test.create_dataset('labels', data = self.test_labels)

    def save_labeled_features(self, filepath, overwrite_flag = False):
        '''
        Method saves a hdf5 file with raw_traces, labels and features:

        TODO: Save the metadata along with the file, so it is documented how the features
        were extracted from the raw traces.
        '''

        new_filepath = filepath
        if new_filepath == self.filename:

            if overwrite_flag == False:
                print(' New filepath will overwrite previous file, please pass overwrite_flag a value of true or provide alternative filepath. Exiting...')
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


    def train_classifier(self, params = {}):
        '''

        Use the hdf5 from the checked data to train and assess the output of the classifier!

        TODO:
        Assert that it has the features etc
        Pass the params dictionary to the random forest?!
        Allow user to choose own classifier
        '''

        self.classifier = NetworkClassifer(self.training_features, self.train_labels, self.test_features, self.test_labels)
        # should throw in the params here!
        self.classifier.run()

        self.classifier.pca(n_components = 3)
        self.classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
        self.classifier.lda_run()
        self.classifier.pca_run()

    def save_classifier(self, savepath):
        f = open(savepath,'wb')
        pickle.dump(self.classifier, f)

    def load_classifier(self, clf_path):
        self.classifier = pickle.load(open(clf_path,'rb'))

    def tune_classifier(self):
        '''
        Pass in dictionary of parameters over which to sweep
        Choose the type of search - random or
        Scoring

        Returns:
            Models according to performance.

        '''
        print('TODO - work on this!')

    def score_classifier(self):
        '''
        Scores classifier on test set
        Uses roc_auc and precision-recall.
        Also prints a classification report.
        '''
        self.classifier.score()

    @ staticmethod
    def _normalise(series):
        a = np.min(series, axis=1)
        b = np.max(series, axis=1)
        return np.divide((series - a[:, None]), (b-a)[:,None])



    ############ This needs to be split into making pdfs in general for the training#####
    ####  And the make training hdf5 needs to be one per file, and then you append to the training dataset ####
    #### Need to have something that lets you exclude traces in the pdf... (if you really really want!)

    def append_to_training(self):
        print("need to implement!")


def main():
    handler = ClassifierHandler()
    handler.load_labeled_traces(filename = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5')

    #handler.correct_labels([(650,3)], 'train')
    handler.check_trace_labels(pdf_savepath = '/Volumes/LACIE SHARE/VM_data/jonny_playing/'+'norm_filtered_',
                               range=(45,75,'train'))
    #handler.save_labelled_traces(filepath='/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_20160223_corrected.hdf5', overwrite_flag=False)

    #handler.extract_features()
    #handler.save_labeled_features('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_201603_18_features.hdf5')
    #handler.load_labeled_features('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/classifier_test_train_201603_18_features.hdf5')
    #handler.train_classifier()
    #handler.save_classifier('/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/pickled_classifier_20160318')

if __name__ == "__main__":
    main()