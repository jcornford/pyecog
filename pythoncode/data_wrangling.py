from __future__ import print_function
import h5py
import os

import numpy as np
import pandas as pd

from converter import NDFLoader
#from make_pdfs import plot_traces_hdf5, plot_traces

class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
    - how we handle the csv filepairs is changing, the annotated dataset will be a tags
    and labels. Do not assign the training or test.
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

    # CSV related handling....
    def make_figures_for_labeling_ndfs(self, path, fs = 512, timewindow = 5,
                                       save_directory = None, format = 'pdf'):
        '''
        Args:
            path: converted ndf path, not directory at the moment
            save_directory: where to put the figures

        Makes figures in supplied savedirectory, or makes a folder above

        '''
        # Sort out save for all of the files
        if save_directory is None:
            savedir = os.path.join(os.path.dirname(os.path.dirname(path)), 'figures_for_labelling')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        data = self._get_converted_h5py_data(path)
        data_array = self._make_array_from_data(data, fs= fs, timewindow=5)

        # make folders within the figures for labelling
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

    def make_annotated_dataset(self, filepairs, savename, fs = 512, timewindow = 5):

        '''
        filepairs: a tuple, or list of tuples in the
        format [(labels.csv, converted_ndf_path)]

        TODO: Change to have the saved the correct format
        '''

        data_array_list = []
        label_list = []

        if type(filepairs) is not list:
            filepairs = [filepairs]

        for pair in filepairs:
            labels = np.loadtxt(pair[0],delimiter=',')[:,1]

            converted_ndf_path = pair[1]
            data = self._get_converted_h5py_data(converted_ndf_path)
            data_array = self._make_array_from_data(data, fs=512,timewindow=5)

            data_array_list.append(data_array)
            label_list.append(labels)

        data_array = np.vstack(data_array_list)
        print(str(data_array.shape)+ ' is shape of dataset ')
        labels = np.hstack(label_list)
        print(str(labels.shape)+ ' is number of labels')

        #Save with flexible shape
        with h5py.File(savename, 'w') as f:
            training = f.create_group('training')
            training.create_dataset('data', data = data_array, maxshape = (None, data_array.shape[1]))
            training.create_dataset('labels', data = labels[:, None], maxshape = (None, 1))

    def append_to_annotated_dataset(self, dataset_path, filepairs, set_type='train', fs = 512, timewindow = 5):
        '''
        This is using CSV and converted ndf filepaths:
        Again, change to have saved in the correct format. Actually not appending in the same way at all.
        Should change this to just adding another group/ustamp_id
        '''
        data_array_list = []
        label_list = []

        if type(filepairs) is not list:
            filepairs = [filepairs]

        for pair in filepairs:
            labels = np.loadtxt(pair[0],delimiter=',')[:,1]
            converted_ndf_path = pair[1]
            data = self._get_converted_h5py_data(converted_ndf_path)
            data_array = self._make_array_from_data(data, fs=512,timewindow=5)

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
                    labels_dset[original_shape[0]:,:] = labels[:, None]

    def make_prediction_dataset(self, converted_ndf_path, savedir = None, fs = 512, timewindow = 5, verbose = False):
        '''
        Args:
            converted_ndf_path:
            fs:
            timewindow:
        '''
        if savedir is None:
            savedir = os.path.join(os.path.dirname(os.path.dirname(converted_ndf)), 'to_predict')

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        data = self._get_converted_h5py_data(converted_ndf_path)
        data_array = self._make_array_from_data(data, fs = 512, timewindow = 5)

        # Write it up! #this isnt going to work at the moment
        savename = os.path.join(savedir, ndfkey+'_id_'+str(tid)+'_pred_db.hdf5')
        print(savename)
        with h5py.File(savename, 'a') as f:
            group = f.create_group(ndfkey+'_id_'+str(tid))
            group.create_dataset('data', data = data_array)

    def make_annotated_set_from_df(self, df, file_dir, fs = 512, timewindow = 5):
        '''
        Pass in pandas dataframe, and the converted ndf directory

        The annotation will be inccorect based on the timewindow coarseness!
        Currently finding the start by start/timewindom -- end/timewindow
        remember with floor division

        Labels only seizure (1) and non seizure(0)

        '''
        annotated_database_name = os.path.dirname(file_dir)+'annotated_database_need_to_add_tag'

        converted_names = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
        converted_tags = [os.path.split(f)[1].split('.')[0] for f in converted_names]
        n_converted = len(converted_tags)

        seizures = []
        count = 0
        for row in df.iterrows():
            fname = row[1].NDF
            name = fname.split('.')[0]
            if name in converted_tags:
                start = row[1]['Seizure Start']
                end = row[1]['Seizure End']
                s_path = os.path.join(file_dir,name+'.hdf5')
                seizures.append({'fname': s_path, 'start': start, 'end': end})
                count += 1
        print('of the '+str(n_converted)+' converted ndfs, '+str(count)+' were overlapped with the dataframe')

        for entry in seizures:
            savedir = self._make_labelled_array_from_seizure_dict(entry, fs, timewindow)

        #self._group_into_test_train_set(savedir, test_size = 0.5)
        self._bundle_labelled_array_files(savedir)

    def _bundle_labelled_array_files(self, file_directory, bundle_name= 'bundled_annotations.h5py', calculate_features = True, filter_order = 3, filter_window = 7):
        '''
        makes a single hdf5 file with traces, labels, and potentially features!
        '''
        files = self._fullpath_listdir(file_directory)
        save_dir = os.path.dirname(file_directory)

        bundle_fpath = os.path.join(save_dir,bundle_name)
        print(bundle_fpath)
        print(str(len(files))+' files to bundle up')
        with h5py.File(bundle_fpath, 'w') as bf:
            # make it after lunch...!
            for f in files:
                data, labels = self._get_dataset_h5py_contents(f)
                ustamp = os.path.split(f)[1].split('.')[0]
                print(data.shape, ustamp)
                ## Now build db file.!! levae test train for later


    @staticmethod
    def _get_dataset_h5py_contents(f):
        '''
        This is used on the grouping method, the saved h5py
        '''
        with h5py.File(f, 'r') as hf:
            assert len(hf.keys()) == 1
            #print(hf.keys())
            group = hf.get(hf.keys()[0])

            data = np.array(group['data'])
            labels = np.array(group['labels'])
        return data, labels

    def _make_labelled_array_from_seizure_dict(self, sdict, fs, timewindow, savedir = None):
        '''
        Method grabs data from the flat converted ndf h5py, and uses the sdict to label
        correctly.

        At this point I should add the feature extraction.

        '''
        if savedir is None:
            savedir = os.path.join(os.path.dirname(os.path.dirname(sdict['fname'])), 'df_generated_annotated')
            #print(savedir)

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        data = self._get_converted_h5py_data(sdict['fname'])
        data_array = self._make_array_from_data(data, fs = 512, timewindow = 5)

        # now use the start and end times to make labels
        labels = np.zeros(shape = (data_array.shape[0]))
        start_i = sdict['start']/5
        end_i = sdict['end']/5

        # Save file
        savename = os.path.join(savedir, os.path.split(sdict['fname'])[1])

        # if we are relabeling an exisiting file (two seizures)
        if os.path.exists(savename):
            with h5py.File(savename, 'r+') as f:
                labels =  f['annotated/labels']
                labels[start_i:end_i] = 1
                print('2nd label round', savename)
        # else the file hasn't been made yet
        else:
            with h5py.File(savename, 'a') as f:
                labels[start_i:end_i] = 1
                group = f.create_group('annotated')
                group.create_dataset('data', data = data_array)
                group.create_dataset('labels', data = labels[:, None])
        return savedir # so the join method knows where to find them

    @staticmethod
    def _make_array_from_data(data, fs, timewindow):
        n_traces = data.shape[0] / (fs * timewindow)
        dp_lost =  data.shape[0] % (fs * timewindow)
        if dp_lost > 0:
            data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))
        else:
            data_array = np.reshape(data, newshape = (n_traces, (fs * timewindow)))
        return data_array

    @staticmethod
    def _get_converted_h5py_data(f):
        '''
        pulling out data only here
        only handling a flat structure at the moment, therefore asserts
        '''
        with h5py.File(f, 'r') as hf:

            assert len(hf.keys()) == 1
            for timestamp in hf.keys():
                M_stamp = hf.get(timestamp)

                assert len(M_stamp.keys()) == 1
                for tid in M_stamp.keys():
                    data = M_stamp.get(tid)['data']

                    return np.array(data)

    def group_into_test_train_set(self, savedir, test_size = 0.25):
        files = self._fullpath_listdir(savedir)
        n_files = len(files)

        data_array_list = []
        label_list = []
        for f in files:
            data, labels = self._get_dataset_h5py_contents(f)
            data_array_list.append(data)
            label_list.append(np.ravel(labels))

        index = int(n_files*test_size)
        #print(index, n_files)
        test = np.vstack(data_array_list)
        train_data_array = np.vstack(data_array_list[:index])
        train_labels = np.hstack(label_list[:index])

        test_data_array = np.vstack(data_array_list[index:])
        test_labels = np.hstack(label_list[index:])
        print('testing: ',test_data_array.shape)
        print('training:',train_data_array.shape)
        print('shouldbe', test.shape)

        filename = os.path.join(os.path.dirname(os.path.dirname(savedir)), 'grouped_annotated_db.hdf5')
        with h5py.File(filename,'w') as f:
            group = f.create_group('training')
            group.create_dataset('data', data = train_data_array)
            group.create_dataset('labels', data = train_labels)

            group = f.create_group('test')
            group.create_dataset('data', data = test_data_array)
            group.create_dataset('labels', data = test_labels)







    @staticmethod
    def _fullpath_listdir(d):
        return [os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')]

    @ staticmethod
    def _normalise(series):
        a = np.min(series, axis=1)
        b = np.max(series, axis=1)
        return np.divide((series - a[:, None]), (b-a)[:,None])


def main():
    '''
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
    #handler.make_prediction_dataset('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/converted_ndfs/M1453393011.hdf5', verbose=True)
    #handler.make_prediction_dataset(os.path.join(dir_n, fn))
    '''
    dname = '/Users/Jonathan/Desktop/temp_nc/'
    df = pd.read_excel(dname+'Animal 93.14.xlsx', sheetname=2)

    handler = DataHandler()
    #handler.convert_ndf(dname+'NDF/', ids=14)
    dir_n = dname+'converted_ndfs'
    handler.make_annotated_set_from_df(df,dir_n )

    #handler.group_into_test_train_set(savedir='/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/df_generated_training/',test_size=0.5)

    #td = '/Volumes/LaCie/Albert_ndfs/Data_03032016/for_testing/M1453331811.hdf5'
    #handler.get_converted_h5py_data(td)


if __name__ == "__main__":
    main()