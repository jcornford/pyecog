import sys
import os
import multiprocessing
import shutil

import h5py
import numpy as np
import pandas as pd

from pyecog.ndf.converter import NDFLoader
from extractor import FeatureExtractor
from utils import filterArray


if sys.version_info < (3,):
    range = xrange
#from make_pdfs import plot_traces_hdf5, plot_traces
class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
    - how we handle the csv filepairs is changing, the annotated dataset will be a tags
    and labels. Do not assign the training or test at this point.

    - csv loading needs to output into the same bundled hdf5 format (tag/ data,labels,features)
     - csv needs overhaul, no need for the list?

    - Correct id handling on converter, the converted file should have stamp & id on filename, name
    not a key

    - handle the unix timestamps better in general, not converting.

    - speed back up the converter!

    - make the converter parallel process directories

    - work out where the feature extraction should take place... currently when bundling the
      labelled arrays.

    '''

    def __init__(self, fs):
        self.fs = fs


    def parallel_ndf_converter(self, ndf_path, t_id,savedir ):
        self.parallel_t_id = t_id
        self.parallel_savedir = savedir

        files = [os.path.join(ndf_path, fname) for fname in os.listdir(ndf_path) if not fname.startswith('.')]
        n_cores = multiprocessing.cpu_count() -1 # so have some freedom!
        pool = multiprocessing.Pool(n_cores)
        pool.map(self._convert_ndf, files)
        pool.close()
        pool.join()

    def convert_ndf_dir(self, path, t_id, savedir = None):
        '''

        Args:
            path: Either a filename or a directory
            ids :
            savedir: If None, will automatically make directory at the same level as the path.
        Returns:

        TODO: Implement fs argument.
        '''
        if os.path.isdir(path):
            print("Reading directory : " + path, ' id : '+ str(t_id))
            filenames = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.ndf')]

            # get save dir if not supplied
            if savedir is None:
                savedir = os.path.join(os.path.split(path[:-1])[0], 'converted_ndfs')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            for filename in filenames:

                self._convert_ndf(filename, t_id, savedir)

        else:
            print("Please enter a valid filepath or directory")
            return 0
        '''
        elif os.path.isfile(path):
            print("Converting file: "+ path, 'id :'+str(ids))

            # get save dir if not supplied
            if savedir is None:
                savedir = os.path.join(os.path.dirname(os.path.dirname(path)), 'converted_ndfs')
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            self._convert_ndf(path, t_id, savedir)
        '''

    def _convert_ndf(self,filename, tid = None, savedir = None):
        # None is when we arre going parallel
        if tid is None:
            tid = self.parallel_t_id
        if savedir is None:
            savedir = self.parallel_savedir

        mname = os.path.split(filename)[1]
        tstamp = float(mname.strip('M').split('.')[0])
        ndf_time = '_'+str(pd.Timestamp.fromtimestamp(tstamp)).replace(':', '_')
        ndf_time =  ndf_time.replace(' ', '_')
        ndf = NDFLoader(filename)
        ndf.load(read_id = tid )
        ndf.glitch_removal(tactic = 'big_guns')
        ndf.correct_sampling_frequency(resampling_fs = self.fs)
        abs_savename = os.path.join(savedir, filename.split('/')[-1][:-4]+'_tid'+str(tid)+ndf_time)
        ndf.save(save_file_name= abs_savename, file_format= 'hdf5')

    # CSV related handling....
    def make_figures_for_labeling_ndfs(self, path, timewindow = 5,
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
        data_array = self._make_array_from_data(data, fs= self.fs, timewindow=5)

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

    def make_annotated_dataset_from_csv(self, filepairs, savename, timewindow = 5):

        '''
        WARNING : SCHEDULED FOR REMOVAL/MERGEMENT?
        Think going to swtich to the 'DF' way of loading things up...
        Still good to know how to do the fexlible stuff though..

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
            data_array = self._make_array_from_data(data, fs=self.fs,timewindow=5)

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

    def append_to_annotated_dataset(self, dataset_path, filepairs, set_type='train', timewindow = 5):
        '''
        WARNING : SCHEDULED FOR REMOVAL/MERGEMENT?
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
            data_array = self._make_array_from_data(data, fs=self.fs,timewindow=5)

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

    @staticmethod
    def get_converted_h5py_features(f):
        '''
        pulling out data only here
        only handling a flat structure at the moment, therefore asserts
        '''

        with h5py.File(f, 'r') as f:
            assert len(f.keys()) == 1 # only one "file" in the h5py file
            timestamp =  list(f.keys())[0]
            M_stamp = f.get(timestamp)

            assert len(M_stamp.keys()) == 1 # only one tid
            tid = list(M_stamp.keys())[0]
            group = M_stamp[tid]
            data = np.array(group['features'])
            return data

    @staticmethod
    def _get_converted_h5py_data(f):
        '''
        pulling out data only here
        only handling a flat structure at the moment, therefore asserts
        '''

        with h5py.File(f, 'r') as f:
            assert len(f.keys()) == 1 # only one "file" in the h5py file
            timestamp =  list(f.keys())[0]
            M_stamp = f.get(timestamp)

            assert len(M_stamp.keys()) == 1 # only one tid
            tid = list(M_stamp.keys())[0]
            group = M_stamp[tid]
            data = np.array(group['data'])
            return data

    def parallel_add_prediction_features(self, h5py_folder):
        files_to_add_features = [os.path.join(h5py_folder, fname) for fname in os.listdir(h5py_folder) if fname.startswith('M')]
        n_cores = multiprocessing.cpu_count() -1 # so have some freedom!
        pool = multiprocessing.Pool(n_cores)
        pool.map(self.add_predfeatures_to_h5py_file, files_to_add_features)
        pool.close()
        pool.join()

    def add_predfeatures_to_h5py_file(self, converted_ndf_path, savedir = None, timewindow = 5, verbose = False,
                                     filter_window = 7, filter_order = 3):
        '''
        Simple: adds features to the converted ndf file. (to be used for predicitons...)
        Args:
            converted_ndf_path: is this a single file?
            fs:
            timewindow:

        TODO :
        '''
        data = self._get_converted_h5py_data(converted_ndf_path)
        data_array = self._make_array_from_data(data, fs = self.fs, timewindow = timewindow)

        try:
            assert data_array.shape[0] == int(3600/timewindow)
        except:
            print('Warning: Data file does not contain, full data: ' +str(os.path.basename(converted_ndf_path))+str(data_array.shape))
            if data_array.shape[0] == 0:
                print('Data file does not contain any data. Exiting: ')
                return 0
        '''
        try:
            assert data_array.all() != np.NaN
        except:
            print('Data is just NaN, exiting...')
            print (data_array)
            return 0
        '''
        #print (data_array)

        fdata = filterArray(data_array, window_size= filter_window, order= filter_order)

        fndata = self._normalise(fdata)

        extractor = FeatureExtractor(fndata, verbose_flag = False)
        features = extractor.feature_array

        # Now add to the converted file!
        with h5py.File(converted_ndf_path, 'r+') as f:
            assert len(f.keys()) == 1 # only one "file" in the h5py file
            timestamp =  list(f.keys())[0]
            M_stamp = f.get(timestamp)

            assert len(M_stamp.keys()) == 1 # only one tid
            tid = list(M_stamp.keys())[0]
            group = M_stamp[tid]
            try: # easy way to allow saving over
                if group['features']:
                    del group['features']
            except KeyError:
                pass
            group.create_dataset('features',data = features)
            if verbose:
                print('Added features to '+str(os.path.basename(converted_ndf_path))+', shape:'+str(features.shape))

    def append_to_seizure_library(self, df, h5_file_dir, existing_seizure_lib,
                                  timewindow = 5,verbose = False):
        '''
        Row needs to have NDF, Seizure Start,Seizure End as headings
        Again, change to have saved in the correct format. Actually not appending in the same way at all.
        Should change this to just adding another group/ustamp_id

        Not sure what is going on when there are no seizures
        '''
        converted_filenames = [os.path.split(f)[1] for f in os.listdir(h5_file_dir) if f.startswith('M')]
        # converted_mocde should be the same as the annotated names (just the stuff at the start)
        converted_mcode = [os.path.split(f)[1][:11] for f in os.listdir(h5_file_dir) if f.startswith('M')]
        n_converted = len(converted_filenames)
        if verbose:
            print (converted_mcode)

        annotations = [] # list to hold files to be bundled
        count = 0
        seizure_tags = []
        for row in df.iterrows():
            annotated_name = str(row[1].NDF).split('.')[0]+'_tid'+str(int(row[1].tid))
            # Now check if we have a match...
            for h5py_name in converted_filenames:
                if h5py_name.startswith(annotated_name):
                    start = row[1]['Seizure Start']
                    end = row[1]['Seizure End']
                    annotations.append({'fname': os.path.join(h5_file_dir,h5py_name), 'start': start, 'end': end})
                    count += 1
                    seizure_tags.append(annotated_name)

        print('Of the '+str(n_converted)+' converted ndfs in directory, '+str(count)+' were found in the passed dataframe')
        tempory_directory = os.path.join(os.path.dirname(h5_file_dir),'temp_dir')
        print ('Creating tempory directory to store labelled array files:...'+tempory_directory)

        if not os.path.exists(tempory_directory):
            os.makedirs(tempory_directory)
        for entry in annotations:
            # entry looks like {'end': 452.4, 'fname': 'SLNDFs/M1457146830.hdf5', 'start': 423.75}
            # timewindow needs to be refactored is len in seconds
            self._make_labelled_array_from_seizure_dict(entry, self.fs, timewindow, tempory_directory)

        try:
            self._bundle_labelled_array_files(tempory_directory, existing_seizure_lib, h5_string = 'r+')
            print ('Cleaning up tempory directory')
            shutil.rmtree(tempory_directory)
        except ValueError:
            print ('Error: '+ str(sys.exc_info()[1]))
            shutil.rmtree(tempory_directory)
            print ('*********** HALTED *************')



    def make_seizure_library_from_df(self, df, h5_file_dir, timewindow = 5,
                                   output_name = 'seizure_library', verbose = False ):
        '''
        WARNING : RENAME? MAKE SEIZURE LIBRARY?
        Pass in pandas dataframe, and the converted ndf directory
        Row needs to have "NDF", "Seizure Start","Seizure End" as headings
        The annotation will be inccorect based on the timewindow coarseness!
        Currently finding the start by start/timewindom -- end/timewindow
        remember with floor division

        Labels only seizure (1) and non seizure(0)

        Makes a "seizure library" bundled file...
         Not sure what is going on when there are no seizures

        '''
        annotated_database_name = os.path.dirname(h5_file_dir)+'annotated_database_need_to_add_tag'

        converted_fullnames = [os.path.join(h5_file_dir, f) for f in os.listdir(h5_file_dir)]
        converted_filenames = [os.path.split(f)[1] for f in os.listdir(h5_file_dir) if f.startswith('M')]
        converted_mcode = [os.path.split(f)[1][:11] for f in os.listdir(h5_file_dir) if f.startswith('M')]
        # converted_mocde should be the same as the annotated names
        # converted tags are the M145etc (the full filename)
        n_converted = len(converted_filenames)
        if verbose:
            print (converted_mcode)

        annotations = []
        count = 0
        seizure_tags = []
        # this is going the wrong way around - should be
        ### should make this different
        for row in df.iterrows():
            #fname = row[1].NDF
            annotated_name = str(row[1].NDF).split('.')[0]+'_tid'+str(int(row[1].tid))
            #print (annotated_name)
            #s_path = os.path.join(file_dir,name+'.hdf5')
            for h5py_name in converted_filenames:
                if h5py_name.startswith(annotated_name):
                    start = row[1]['Seizure Start']
                    end = row[1]['Seizure End']
                    annotations.append({'fname': os.path.join(h5_file_dir,h5py_name), 'start': start, 'end': end})
                    count += 1
                    seizure_tags.append(annotated_name)

        # Do i need this stuff? # no think will assume the whole folder has been annotATED
        # GIVEN you upsample, don't worry about it?
        '''
        non_seizures = [tag for tag in converted_mcode if tag not in seizure_tags]
        for name in non_seizures:
            print
            for h5py_name in converted_filenames:
                if h5py_name.startswith(name):
                    s_path = os.path.join(h5_file_dir,h5py_name)
            start = 0
            end = 0
            annotations.append({'fname': s_path, 'start': start, 'end': end})
       '''

        print('Of the '+str(n_converted)+' converted ndfs in directory, '+str(count)+' were found in the passed dataframe')
        tempory_directory = os.path.join(os.path.dirname(h5_file_dir),'temp_dir')
        print ('Creating tempory directory to store labelled array files:...'+tempory_directory)

        if not os.path.exists(tempory_directory):
            os.makedirs(tempory_directory)
        for entry in annotations:
            # entry looks like {'end': 452.4, 'fname': 'SLNDFs/M1457146830.hdf5', 'start': 423.75}
            # timewindow needs to be refactored is len in seconds
            self._make_labelled_array_from_seizure_dict(entry, self.fs, timewindow, tempory_directory)

        self._bundle_labelled_array_files(tempory_directory, output_name, h5_string = 'w')
        print ('Cleaning up tempory directory')
        shutil.rmtree(tempory_directory)
    def _bundle_labelled_array_files(self, file_directory, library_file,
                                     calculate_features = True, filter_order = 3, filter_window = 7,
                                    h5_string = 'w'):
        '''
        Unclear file and where it fits... bundle labelled arra_file into seizure librry?
        Hoping to use for both first time and also adding to the libraries...!
        makes a single hdf5 file with traces, labels, and  features

        file_directory? where all the annotated array files are.
        output name is the seizure lib will be

        Maybe ustamps are going to need to contain t_id...!
        '''
        files = self._fullpath_listdir(file_directory)
        print(str(len(files))+' files to bundle up from'+file_directory)

        with h5py.File(library_file, h5_string) as bf:

            for f in files:
                data, labels = self._get_dataset_h5py_contents(f)
                ustamp = os.path.split(f)[1].split('.')[0]
                print('Adding',ustamp, data.shape, 'to',library_file )
                ## Now build db file.!! levae test train for later
                group = bf.create_group(ustamp)
                group.create_dataset('data', data = data)
                group.create_dataset('labels', data = labels)

                if calculate_features:
                    fdata = filterArray(data, window_size= filter_window, order= filter_order)
                    fndata = self._normalise(fdata)
                    extractor = FeatureExtractor(fndata)
                    features = extractor.feature_array
                    group.create_dataset('features', data = features)

    @staticmethod
    def _get_dataset_h5py_contents(f):
        '''
        This is used on the grouping method, the saved h5py
        '''
        with h5py.File(f, 'r') as hf:
            assert len(hf.keys()) == 1
            for key in hf.keys():
                group = hf.get(key)
            data = np.array(group['data'])
            labels = np.array(group['labels'])
        return data, labels

    def _make_labelled_array_from_seizure_dict(self, sdict, fs, timewindow, temp_dir, verbose = False):
        '''
        Inputs:
            - sdict is entry from df,:
            - fs,
            - timewindow
            - tempory_directory : where labelled arrays are to be stored

        Method grabs data from the flat converted ndf h5py, and uses the sdict to label
        correctly.

        At this point I should add the feature extraction? Currently added on the bundling bit.

        Then  temp directory is used by the bundler..

        '''

        data = self._get_converted_h5py_data(sdict['fname'])
        data_array = self._make_array_from_data(data, fs = fs, timewindow = timewindow)
        # now use the start and end times to make labels
        labels = np.zeros(shape = (data_array.shape[0]))
        start_i = int(np.floor(sdict['start']/5.0))
        end_i = int(np.ceil(sdict['end']/5.0))
        if verbose:
            print (start_i, sdict['start'], end_i,sdict['end'], sdict['fname'], data_array.shape)

        # Save file
        savename = os.path.join(temp_dir, os.path.split(sdict['fname'])[1])

        # if we are relabeling an exisiting file (two seizures)
        if os.path.exists(savename):
            with h5py.File(savename, 'r+') as f:
                labels =  f['annotated/labels']
                labels[start_i:end_i] = 1
                print(savename,'has more than one seizure!')
        # else the file hasn't been made yet
        else:
            with h5py.File(savename, 'a') as f:
                labels[start_i:end_i] = 1
                group = f.create_group('annotated')
                group.create_dataset('data', data = data_array)
                group.create_dataset('labels', data = labels[:, None])

    @staticmethod
    def _make_array_from_data(data, fs, timewindow):
        n_traces = int(data.shape[0] / (fs * timewindow))
        dp_lost =  int(data.shape[0] % (fs * timewindow))
        if dp_lost > 0:
            data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, int(fs * timewindow)))
        else:
            data_array = np.reshape(data, newshape = (n_traces, int(fs * timewindow)))
        return data_array


    @staticmethod
    def _fullpath_listdir(d):
        return [os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')]

    @ staticmethod
    def _normalise(series):

        a = np.min(series, axis=1)
        b = np.max(series, axis=1)

        try:
            assert (b-a).all() != 0.0
            result = np.divide((series - a[:, None]), (b-a)[:,None])
            return result
        except:
            print ('Warning: Zero div error caught, exiting. Items all the same')
            import sys
            sys.exit()