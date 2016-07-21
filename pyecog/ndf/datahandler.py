import sys
import os
import multiprocessing
import shutil
import traceback
import time

import h5py
import numpy as np
import pandas as pd

from .converter import NdfFile
from .extractor import FeatureExtractor
from .utils import filterArray

if sys.version_info < (3,):
    range = xrange
#from make_pdfs import plot_traces_hdf5, plot_traces
class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
     - Generate training/seizure library using ndfs and df (bundle directly - many datasets??)
     - generate single channel folders for predictions, pass in dates, and print out files you skipped

     - use the h5file and ndfFile class to clean up the code (use indexing for future compat with h5 or ndf)

    To think about
    - handle the unix timestamps better in general
    - enable feature extraction to be better handled
    - filtering and pre processing
    - remeber when defining maxshape = (None, data_array.shape[1])) - to allow h5 to gorwp

    '''

    def __init__(self):

        self.parallel_savedir = None

    @staticmethod
    def _get_dataset_h5py_contents(f):
        '''
        Redundant!
        This is used on the grouping method, the saved h5py
        '''
        with h5py.File(f, 'r') as hf:
            assert len(hf.keys()) == 1
            for key in hf.keys():
                group = hf.get(key)
            data = np.array(group['data'])
            labels = np.array(group['labels'])
        return data, labels


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

    def convert_ndf_directory_to_h5(self, ndf_dir, tids = 'all', save_dir  = 'same_level', n_cores = -1, fs = 'auto'):
        """

        Args:
            ndf_dir: Directory to convert
            tids: transmitter ids to convert. Default is 'all'
            save_dir: optional save directory, will default to appending convertered_h5s after current ndf
            n_cores: number of cores to use
            fs :  'auto' or frequency in hz

        ndfs conversion seem to be pretty buggy...

        """
        self.fs_for_parallel_conversion = fs
        files = [f for f in self._fullpath_listdir(ndf_dir) if f.endswith('.ndf')]

        # check ids
        ndf = NdfFile(files[0])
        if not tids == 'all':
            if not hasattr(tids, '__iter__'):
                tids = [tids]
            for tid in tids:
                try:
                    assert tid in ndf.tid_set
                except AssertionError:
                    print('Please enter valid tid (at least for the first!) file in directory ('+str(ndf.tid_set)+')')
                    sys.exit()

        self.tids_for_parallel_conversion = tids
        print ('Transmitters for conversion: '+ str(self.tids_for_parallel_conversion))

        # set n_cores
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()

        # Make save directory
        if save_dir  == 'same_level':
            save_dir = ndf_dir+'_converted_h5s'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.savedir_for_parallel_conversion = save_dir

        pool = multiprocessing.Pool(n_cores)
        pool.map(self._convert_ndf, files)
        pool.close()
        pool.join()

    def _convert_ndf(self,filename):
        savedir = self.savedir_for_parallel_conversion
        tids = self.tids_for_parallel_conversion
        fs = self.fs_for_parallel_conversion

        # convert m name
        mname = os.path.split(filename)[1]
        tstamp = float(mname.strip('M').split('.')[0])
        ndf_time = '_'+str(pd.Timestamp.fromtimestamp(tstamp)).replace(':', '_')
        ndf_time =  ndf_time.replace(' ', '_')
        start = time.time()
        try:
            ndf = NdfFile(filename, fs = fs)
            if set(tids).issubset(ndf.tid_set) or tids == 'all':
                ndf.load(tids)
                abs_savename = os.path.join(savedir, filename.split('/')[-1][:-4]+ndf_time+' tids_'+str(tids))
                ndf.save(save_file_name= abs_savename)
            else:
                print('not all tids:'+str(tids) +' were valid for '+str(os.path.split(filename)[1])+' skipping!')

        except Exception:
            print('Something went wrong loading '+str(tids)+' from '+mname+' :')
            #print('Valid ids are:'+str(ndf.tid_set))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print (traceback.print_exception(exc_type, exc_value,exc_traceback))
        return 0 # don't think i actually this
