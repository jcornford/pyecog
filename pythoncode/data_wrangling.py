from __future__ import print_function
import h5py
import os

import numpy as np

from converter import NDFLoader
from make_pdfs import plot_traces_hdf5, plot_traces

class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
    - refactor h5py handling into a new function
    - why do we always lose three datapoints
     - Smooth out id handling on converter and fs
     - handle the unix timestamps better.
     - speed back up the converter!
     - make the converter parallel process directories.

    '''

    def __init__(self):
        pass

    def convert_ndf(self, path, ids = -1, savedir = None):
        '''

        Args:
            path: Either a filename or a directory
            ids :
            savedir: If None, will automatically make directory at the same level as the path.
        Returns:

        TODO: Implement fs argument.
        '''
        if os.path.isdir(path):
            print("Reading directory : " + path, ' id : '+ str(ids))
            filenames = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.ndf')]

            # get save dir if not supplied
            if savedir is None:
                savedir = os.path.join(os.path.split(path[:-1])[0], 'converted_ndfs')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            for filename in filenames:
                self._convert_ndf(filename, ids, savedir)

        elif os.path.isfile(path):
            print("Converting file: "+ path, 'id :'+str(ids))

            # get save dir if not supplied
            if savedir is None:
                savedir = os.path.join(os.path.dirname(os.path.dirname(path)), 'converted_ndfs')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            self._convert_ndf(path, ids, savedir)

        else:
            print("Please enter a valid filepath or directory")
            return 0

        if not os.path.exists(savedir):
            os.makedirs(savedir)

    @staticmethod
    def _convert_ndf(filename, ids, savedir):
        ndf = NDFLoader(filename)
        ndf.load([ids])
        ndf.glitch_removal(read_id=[ids])
        ndf.correct_sampling_frequency(read_id=[ids], fs = 512.0)
        ndf.save(save_file_name= os.path.join(savedir, filename.split('/')[-1][:-4]), file_format= 'hdf5')

    @ staticmethod
    def _normalise(series):
        a = np.min(series, axis=1)
        b = np.max(series, axis=1)
        return np.divide((series - a[:, None]), (b-a)[:,None])

    def make_figures_for_labeling_ndfs(self,path, fs = 512, timewindow = 5,
                                       save_directory = None, format = 'pdf'):
        '''
        Args:
            path: converted ndf path
            save_directory:

        Makes figures in supplied savedirectory, or makes a folder above

        ToDo: Make individual folders for each ndf
        '''
        # Sort out save directory
        if save_directory is None:
            savedir = os.path.join(os.path.dirname(os.path.dirname(path)), 'figures_for_labelling')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with h5py.File(path , 'r') as hf:
                    for ndfkey in hf.keys():
                        datadict = hf.get(ndfkey)

                    for tid in datadict.keys():
                        data = np.array(datadict[tid]['data'])
                        n_traces = data.shape[0] / (fs * timewindow)

                        print('plotting transmitter id ' + str(tid)+'... There are '+ str(n_traces) + ' traces')
                        dp_lost =  data.shape[0] % (fs * timewindow)

                        data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))
                        data_array = self._normalise(data_array)

                        figure_folder = os.path.join(savedir, path.split('/')[-1].split('.')[0]+'_id'+str(tid))

                        print(figure_folder)
                        if not os.path.exists(figure_folder):
                            os.makedirs(figure_folder)

                        black = np.ones((data.shape[0]))
                        plot_traces_hdf5(data_array,
                                    labels = black,
                                    savepath = figure_folder,
                                    filename = path.split('/')[-1].split('.')[0],
                                    format_string = format,
                                    trace_len_sec = timewindow)

    def append_to_annotated_dataset(self, dataset_path, filepairs,set_type='train', fs = 512, timewindow = 5):

        if type(filepairs) is not list:
            filepairs = [filepairs]

        data_array_list = []
        label_list = []
        for pair in filepairs:
            labels = np.loadtxt(pair[0],delimiter=',')[:,1]
            converted_ndf = pair[1]

            with h5py.File(converted_ndf , 'r') as hf:
                    for ndfkey in hf.keys():
                        datadict = hf.get(ndfkey)

                    for tid in datadict.keys():
                        print('transmitter id ' + str(tid))
                        data = np.array(datadict[tid]['data'])

                        n_traces = data.shape[0] / (fs * timewindow)
                        dp_lost =  data.shape[0] % (fs * timewindow)

                        if dp_lost > 0:
                            print('still')
                            data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))
                        else:
                            print('here')

                            data_array = np.reshape(data, newshape = (n_traces, (fs * timewindow)))

                    data_array_list.append(data_array)
                    label_list.append(labels)

        data_array = np.vstack(data_array_list)
        print(str(data_array.shape)+ ' is shape of dataset to append ')
        labels = np.hstack(label_list)
        print(str(labels.shape)+ ' is number of labels to append')
        assert data_array.shape[0] == labels.shape[0]

        if set_type == 'train':
            with h5py.File(dataset_path, 'r+') as db:
                data_dset =  db['training/data']
                labels_dset =  db['training/labels']

                # resize the dataset
                original_shape =  data_dset.shape
                new_shape = original_shape[0] + data_array.shape[0]
                print('original training shape was: ' + str(original_shape))
                data_dset.resize(new_shape, axis = 0)
                labels_dset.resize(new_shape, axis = 0)
                print('shape is now '+str(data_dset.shape))

                #Add the data
                data_dset[original_shape[0]:,:] = data_array
                labels_dset[original_shape[0]:,:] = labels[:,None]

        if set_type == 'test':
            with h5py.File(dataset_path, 'r+') as db:
                if 'test' not in db.keys():
                    print('Added test dataset to file')
                    test = db.create_group('test')
                    test.create_dataset('data', data = data_array, maxshape = (None, data_array.shape[1]))
                    test.create_dataset('labels', data = labels[:, None], maxshape = (None, 1))
                else:
                    data_dset =  db['test/data']
                    labels_dset =  db['test/labels']
                    original_shape =  data_dset.shape

                    new_shape = original_shape[0] + data_array.shape[0]

                    print('original test shape was: ' + str(original_shape))
                    data_dset.resize(new_shape, axis = 0)
                    labels_dset.resize(new_shape, axis = 0)
                    print('shape is now '+str(data_dset.shape))

                    #Add the data
                    data_dset[original_shape[0]:,:] = data_array
                    labels_dset[original_shape[0]:,:] = labels[:,None]

    def make_prediction_dataset(self, converted_ndf, savedir = None, fs = 512, timewindow = 5, verbose = False):
        '''

        Args:
            fs:
            timewindow:

        Returns:
             - we need a way to keep track of filename...
             1. Either a new dataset per file (easiest)
             2. HDF5 file with keys...
        '''
        if savedir is None:
            savedir = os.path.join(os.path.dirname(os.path.dirname(converted_ndf)), 'to_predict')
            #print(savedir)

        if not os.path.exists(savedir):
            os.makedirs(savedir)


        #print(converted_ndf)
        with h5py.File(converted_ndf, 'r') as hf:
            for ndfkey in hf.keys():
                datadict = hf.get(ndfkey)

                for tid in datadict.keys():


                        data = np.array(datadict[tid]['data'])
                        n_traces = data.shape[0] / (fs * timewindow)

                        dp_lost =  data.shape[0] % (fs * timewindow)

                        if verbose:
                            print('shape', data.shape)
                            print('dp lost', str(dp_lost))
                            print('There are '+ str(n_traces) + ' traces')
                            print('transmitter id ' + str(tid))

                        if dp_lost > 0:
                            data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))
                        else:
                            data_array = np.reshape(data, newshape = (n_traces, (fs * timewindow)))

                        ### Write it up! ###
                        savename = os.path.join(savedir, ndfkey+'_id_'+str(tid)+'_pred_db.hdf5')
                        #print(savename)
                        with h5py.File(savename, 'a') as f:
                            group = f.create_group(ndfkey+'_id_'+str(tid))
                            group.create_dataset('data', data = data_array)



    def make_annotated_dataset(self, filepairs, savename, fs = 512, timewindow = 5):

        '''
        WARNING: CURRENTLY NOT HANDLING MULTPLE TIDS
        filepairs: a tuple, or list of tuples in the
        format [(labels.csv, converted_ndf)]

        '''

        data_array_list = []
        label_list = []

        if type(filepairs) is not list:
            filepairs = [filepairs]

        for pair in filepairs:
            labels = np.loadtxt(pair[0],delimiter=',')[:,1]

            converted_ndf = pair[1]
            with h5py.File(converted_ndf , 'r') as hf:
                    for ndfkey in hf.keys():
                        datadict = hf.get(ndfkey)

                    for tid in datadict.keys():
                        print('transmitter id ' + str(tid))
                        data = np.array(datadict[tid]['data'])

                        n_traces = data.shape[0] / (fs * timewindow)
                        print('There are '+ str(n_traces) + ' traces')
                        dp_lost =  data.shape[0] % (fs * timewindow)

                        data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))

                    data_array_list.append(data_array)
                    label_list.append(labels)


        data_array = np.vstack(data_array_list)
        print(str(data_array.shape)+ ' is shape of dataset ')
        labels = np.hstack(label_list)
        print(str(labels.shape)+ ' is number of labels')

        ### Write it up! ###
        with h5py.File(savename, 'w') as f:

            training = f.create_group('training')
            training.create_dataset('data', data = data_array, maxshape = (None, data_array.shape[1]))
            training.create_dataset('labels', data = labels[:, None], maxshape = (None, 1))



def main():
    handler = DataHandler()
    #handler.convert_ndf('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.8/NDF/', ids = 14)
    #handler.make_figures_for_labeling_ndfs('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.8/converted_ndfs/M1454346216.hdf5', timewindow=5)

    basedir = '/Volumes/LaCie/Albert_ndfs/training_data/raw_hdf5s/'
    filepairs = [('state_labels_2016_01_21_19-16.csv','2016_01_21_19:16.hdf5'),
                     ('state_labels_2016_01_21_13-16.csv','2016_01_21_13:16.hdf5'),
                     ('state_labels_2016_01_21_11-16.csv','2016_01_21_11:16.hdf5'),
                     ('state_labels_2016_01_21_10-16.csv','2016_01_21_10:16.hdf5'),
                     ('state_labels_2016_01_21_08-16.csv','2016_01_21_08:16.hdf5')]
    pair = (os.path.join(basedir,'state_labels_2016_01_21_19-16.csv'), os.path.join(basedir,'2016_01_21_19:16.hdf5'))
    pair2 = (os.path.join(basedir,'state_labels_2016_01_21_13-16.csv'), os.path.join(basedir,'2016_01_21_13:16.hdf5'))
    db_name = '/Volumes/LaCie/Albert_ndfs/training_data/training_data_v2_jonny_playing.hdf5'
    #handler.make_annotated_dataset(pair, db_name)
    #handler.append_to_annotated_dataset(db_name,pair2, set_type='test')

    dir_n = '/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/converted_ndfs'
    fn = 'M1453331811.hdf5'
    fn = 'M1453364211.hdf5'
    handler.make_prediction_dataset('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/converted_ndfs/M1453393011.hdf5', verbose=True)
    #handler.make_prediction_dataset(os.path.join(dir_n, fn))

if __name__ == "__main__":
    main()