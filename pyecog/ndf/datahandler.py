import sys
import os
import multiprocessing
import shutil
import traceback
import time
import logging

import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats

from .ndfconverter import NdfFile
from .h5loader import H5File
from .feature_extractor import FeatureExtractor

if sys.version_info < (3,):
    range = xrange

import logging

#from make_pdfs import plot_traces_hdf5, plot_traces
class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
     -  pass in dates for predictions, and print out files you skipped is not transmitter there

    To think about
    - enable feature extraction to be better handled
    - remeber when defining maxshape = (None, data_array.shape[1])) - to allow h5 to be resized


    import multiprocessing as mp
    import time

    def foo_pool(x):
        time.sleep(2)
        return x*x

    result_list = []
    def log_result(result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    def apply_async_with_callback():
        pool = mp.Pool(4)
        for i in range(10):
        pool.imap(foo_pool, args = (i, ), callback = log_result)
        pool.close()
        pool.join()
        print(result_list)

apply_async_with_callback()
    '''

    def __init__(self, logpath = os.getcwd()):

        self.parallel_savedir = None
        self.parrallel_flag_pred = False

    def add_labels_to_seizure_library(self, library_path, overwrite, timewindow):
        '''

        Called by self.add_features_seizure_library()

        '''
        logging.info('Adding labels to '+ library_path)
        # todo check timewindows go into n seconds precisely
        with h5py.File(library_path, 'r+') as f:
            seizure_datasets = [f[group] for group in list(f.keys())]

            for i, group in enumerate(seizure_datasets):
                # First check if there are labels already there. If overwrite, go to except block
                try:
                    if not overwrite:
                        features = group['labels']
                        logging.info(str(group)+' already has labels, skipping')
                    if overwrite:
                        raise
                except:
                    try: # attempt to delete, will throw error if group doesn't exist.
                        del group['labels']
                        del group.attrs['chunked_annotation']

                        logging.debug('Deleted old labels')
                    except:
                        pass
                    # make labels dataset here:
                    # first work out the would-be shape of the data array
                    data = group['data'][:]
                    fs = group.attrs['fs']
                    n_traces = int(data.shape[0] / (fs * timewindow))
                    #dp_lost =  int(data.shape[0] % (fs * timewindow))
                    # dp per chunk is then int(fs * timewindow))
                    #print(n_traces)
                    labels = np.zeros(n_traces)
                    #print(group.attrs['precise_annotation'])
                    group.attrs['chunk_length'] = timewindow
                    for row in group.attrs['precise_annotation']:
                        start_i = int(np.floor(row[0]/timewindow))
                        end_i   = int(np.ceil(row[1]/timewindow))
                        labels[start_i:end_i] = 1

                        try:
                            group.attrs['chunked_annotation'] = np.vstack([group.attrs['chunked_annotation'],
                                                                           np.array([(start_i*timewindow,end_i*timewindow)]) ])
                        except KeyError:
                            group.attrs['chunked_annotation'] = np.array([(start_i*timewindow,end_i*timewindow)])
                    ###
                    #print(group.attrs['chunked_annotation'])


                    group.create_dataset('labels', data = labels, compression = "gzip", dtype = 'i2', chunks = labels.shape)

    def add_features_seizure_library(self, library_path, overwrite = False, run_peaks = True, timewindow = 5):
        '''
        Inputs:
        - overwrite: if false, will only add labels and features to seizures in the library that do not already
          have accompanying labels and features.

        This is painfully slow at the moment, not accessing hdf5 groups in parallel.
        Both adding features to library and prediction h5 file now are expecting unchunked data. This method
        will make labels too though
        '''
        logging.info('First adding labels to ' + library_path)
        print('Adding labels first...', end=' ')
        self.add_labels_to_seizure_library(library_path, overwrite, timewindow)
        print('Done')

        logging.info('Adding features to ' + library_path)
        with h5py.File(library_path, 'r+') as f:
            seizure_datasets = [f[group] for group in list(f.keys())]

            l = len(seizure_datasets)-1
            self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
            for i, group in enumerate(seizure_datasets):
                logging.debug('Loading group: '+ str(group))

                # First check if there are features already there. If overwrite, go to except block
                try:
                    if not overwrite:
                        features = group['features']
                        logging.info(str(group)+' already has features, skipping')
                    if overwrite:
                        raise
                except:
                    data_array = group['data'][:]
                    data_array = self._make_array_from_data(data_array, fs = group.attrs['fs'], timewindow = timewindow)
                    assert len(data_array.shape) > 1

                    if data_array is not None:
                        extractor = FeatureExtractor(data_array, fs = group.attrs['fs'], run_peakdet = run_peaks)
                        features = extractor.feature_array
                        try: # attempt to delete, will throw error if group doesn't exist.
                            del group['features']
                            logging.debug('Deleted old features')
                        except:
                            pass
                        group.create_dataset('features', data = features, compression = 'gzip', dtype = 'f4')
                        group.attrs['col_names'] = np.array(extractor.col_labels).astype('|S9')
                        # here add feature titles to the dataset attrs?
                        logging.info('Added features to ' + str(group) + ', shape:' + str(features.shape))
                        self.printProgress(i,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
                    # Data_array is none
                    else:
                        logging.error("Didn't add features to file: "+str(group))
                        return 0

    def add_predicition_features_to_h5_file(self, h5_file_path, timewindow = 5, run_peakdet = True):
        '''
        Currently assuming only one transmitter per h5 file

        Could eventually (now only one tid) have the structure of:

        M1445612etc / tid1
                    / tid2
                    / tid3  / data
                            / time
        '''
        if self.parrallel_flag_pred:
            run_peakdet = self.run_pkdet
            timewindow  = self.twindow

        with h5py.File(h5_file_path, 'r+') as f:

            mcodes = [f[group] for group in list(f.keys())]
            assert len(mcodes) == 1 # assuming one transmitter per file
            mcode = mcodes[0]

            tids = [mcode[tid] for tid in list(mcode.keys())]

            for tid in tids: # loop through groups which are tids for predicition h5s
                data_array = tid['data'][:]

                if data_array is not None:

                    logging.debug('Reshaping data from '+str(data_array.shape)+ ' using '+str(tid.attrs['fs']) +' fs' +
                                  ' and timewindow of '+ str(timewindow)  )
                    data_array = self._make_array_from_data(data_array, fs = tid.attrs['fs'], timewindow = timewindow)
                    try:
                        assert data_array.shape[0] == int(3600/timewindow)
                    except:
                        print('Warning: Data file does not contain, full data: ' + str(os.path.basename(h5_file_path)) + str(data_array.shape))
                    #return data_array
                    extractor = FeatureExtractor(data_array, tid.attrs['fs'], run_peakdet = run_peakdet)
                    features = extractor.feature_array

                    try:
                        del tid['features']
                        logging.debug('Deleted old features')
                    except:
                        pass
                    tid.create_dataset('features', data = features, compression = 'gzip')
                    tid.attrs['col_names'] = np.array(extractor.col_labels).astype('|S9')

                    logging.debug('Added features to ' + str(os.path.basename(h5_file_path)) + ', shape:' + str(features.shape))

                elif data_array is None:
                    logging.error('File has None for data - perhaps all the same')
                    logging.error("Didn't add features to group: "+ str(tid))

                    return 0

    def parallel_add_prediction_features(self, h5py_folder, n_cores = -1, run_peakdet = False, timewindow = 5):
        '''
        # NEED TO ADD SETTINGS HERE FOR THE TIMEWINDOW ETC

        '''
        self.parrallel_flag_pred = True
        self.run_pkdet = run_peakdet
        self.twindow = timewindow
        files_to_add_features = [os.path.join(h5py_folder, fname) for fname in os.listdir(h5py_folder) if fname.startswith('M')] #switch to self.fullpath_listdir()
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(n_cores)
        l = int(len(files_to_add_features))
        print( ' Adding features to '+str(l)+ ' hours in '+ h5py_folder)
        self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        for i, _ in enumerate(pool.imap(self.add_predicition_features_to_h5_file, files_to_add_features), 1):
            self.printProgress(i,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

        #pool.map(self.add_predicition_features_to_h5_file, files_to_add_features)
        pool.close()
        pool.join()
        self.parrallel_flag_pred = False
    def get_annotations_from_df_datadir_matches(self, df, file_dir):
        '''
        This function matches the entries in a dataframe with files in a directory

        Returns: list of annotations stored in a list
        '''

        abs_filenames = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
        data_filenames = [f for f in os.listdir(file_dir) if f.startswith('M')]
        mcodes = [os.path.split(f)[1][:11] for f in os.listdir(file_dir) if f.startswith('M')]
        n_files = len(data_filenames)

        # now loop through matching the tid to datafile in the annotations
        df.columns = [label.lower() for label in df.columns]
        df.columns  = [label.strip(' ') for label in df.columns]

        reference_count = 0
        annotation_dicts = []
        for row in df.iterrows():
            # annotation name is bad, but will ultimately be the library h5 dataset name
            annotation_name = str(row[1]['filename']).split('.')[0]+'_tid_'+str(int(row[1]['transmitter']))
            for datafile in data_filenames:
                if datafile.startswith(annotation_name.split('_')[0]):
                    start = row[1]['start']
                    end = row[1]['end']
                    annotation_dicts.append({'fname': os.path.join(file_dir, datafile),
                                             'start': start,
                                             'end': end,
                                             'dataset_name': annotation_name,
                                             'tid':int(row[1]['transmitter'])})
                    reference_count += 1

        print('Of the '+str(n_files)+' ndfs in directory, '+str(reference_count)+' references to seizures were found in the passed dataframe')
        return annotation_dicts

    def make_seizure_library(self, df, file_dir,
                             timewindow = 5,
                             seizure_library_name = 'seizure_library',
                             fs = 'auto',
                             verbose = False,
                             overwrite = False,
                             scale_and_filter = False):
        '''
        Args:

            df : pandas dataframe. Column titles need to be "filename", "start","end", "transmitter"
            file_dir: path to converted h5, or ndf directory, that contains files referenced in
                      the dataframe
            timewindow: size to chunk the data up with
            seizure_library_name: path and name of the seizure lib.
            fs: default is auto, but use freq in hz to sidestep the auto dectection
            verbose: Flag, print or not.
            - scale and filter should not be used (for simply keeping in original y units)

        Returns:
            Makes a Seizure library file

        WARNING: The annotation will be incorrect based on the time-window coarseness and the chunk that is chosen!
        Currently finding the start by start/timewindom -- end/timewindow


        TODO:
        -  How to handle files that don't have seiures, but we want to include
        -  Not sure what is going on when there are no seizures, need to have this functionality though.

        '''

        logging.info('Datahandler - creating SeizureLibrary')
        try:
            annotation_dicts = self.get_annotations_from_df_datadir_matches(df, file_dir)
        except:
            print("Error getting annotations from your file. Please ensure columns are named: 'filename', 'transmitter','start','end'")
            annotation_dicts = self.get_annotations_from_df_datadir_matches(df, file_dir)
        # annotations_dicts is a list of dicts with... e.g 'dataset_name': 'M1445443776_tid_9',
        # 'end': 2731.0, 'fname': 'all_ndfs/M1445443776.ndf', 'start': 2688.0,' tid': 9

        h5code = 'w' if overwrite else 'x'
        try:
            if not '/' in seizure_library_name or not "\\" in seizure_library_name:
                seizure_library_path = os.path.split(file_dir)[0]+seizure_library_name.strip('.h5')+'.h5'
            seizure_library_path = seizure_library_name.strip('.h5')+'.h5'
            print('Creating seizure library: '+ seizure_library_path)
            logging.info('Creating seizure library: '+ seizure_library_path)
            h5file = h5py.File(seizure_library_path, h5code)
            h5file.attrs['fs'] = fs
            h5file.attrs['timewindow'] = timewindow
            h5file.close()

        except Exception:
            print ('Error: Seizure library file exists! Delete it or set "overwrite" to True')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print (traceback.print_exception(exc_type, exc_value, exc_traceback))
            return 0

        # now populate to seizure lib with data, time and labels
        # make a list
        l = len(annotation_dicts)-1
        self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        for i, annotation in enumerate(annotation_dicts):
            self._populate_seizure_library(annotation,
                                           fs,
                                           timewindow,
                                           seizure_library_path,
                                           verbose = verbose,
                                           scale_and_filter = scale_and_filter)
            self.printProgress(i,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

    def append_to_seizure_library(self, df, file_dir, seizure_library_path,
                                  timewindow = 5,
                                  fs = 'auto',
                                  verbose = False,
                                  overwrite = False,
                                  scale_and_filter = False):
        '''
        Args:

            df : pandas dataframe. Column titles need to be "name", "start","end", "transmitter"
            file_dir: path to converted h5, or ndf directory, that contains files referenced in
                      the dataframe
            timewindow: size to chunk the data up with
            seizure_library_path: path and name of the seizure lib.
            fs: default is auto, but use freq in hz to sidestep the auto dectection
            verbose: Flag, print or not.

        Returns:
            Appends to a Seizure library file

        WARNING: The annotation will be incorrect based on the time-window coarseness and the chunk that is chosen!
        Currently finding the start by start/timewindom -- end/timewindow


        TODO:
        -  add documentaiton of the loading settings
        -  How to handle files that don't have seiures, but we want to include (done, put to 00's)
        -  Not sure what is going on when there are no seizures, need to have this functionality though.

        '''

        logging.info('Appending to seizure library')
        annotation_dicts = self.get_annotations_from_df_datadir_matches(df, file_dir)
        # annotations_dicts is a list of dicts with... e.g 'dataset_name': 'M1445443776_tid_9',
        # 'end': 2731.0, 'fname': 'all_ndfs/M1445443776.ndf', 'start': 2688.0,' tid': 9

        # now add to to seizure lib with data, time and labels
        l = len(annotation_dicts)-1
        logging.info('Datahandler - creating SeizureLibrary')
        self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        for i, annotation in enumerate(annotation_dicts):
            self._populate_seizure_library(annotation, fs, timewindow, seizure_library_path, verbose = verbose, scale_and_filter = scale_and_filter)
            self.printProgress(i,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)


    def _populate_seizure_library(self, annotation, fs,
                                  timewindow,
                                  seizure_library_path,
                                  verbose = False,
                                  scale_and_filter = False):


        logging.debug('Adding '+str(annotation['fname']))
        if annotation['fname'].endswith('.ndf'):
            file_obj = NdfFile(annotation['fname'],fs = fs)
            file_obj.load(annotation['tid'], scale_to_mode_std= scale_and_filter)
        elif annotation['fname'].endswith('.h5'):
            file_obj = H5File(annotation['fname'])
        else:
            print('ERROR: Unrecognised file-type')

        # add in filtering and scaling here
        #data_array = self._make_array_from_data(file_obj[annotation['tid']]['data'], fs, timewindow)
        data_array = file_obj[annotation['tid']]['data'] # just 1D at the moment!

        # use the start and end times to make labels
        #labels = np.zeros(shape = (data_array.shape[0]))
        #start_i = int(np.floor(annotation['start']/timewindow))
        #end_i   = int(np.ceil(annotation['end']/timewindow))

        #print('  ')
        #print(annotation['start'], ': ', start_i*timewindow)
        #print(annotation['end'], ':', end_i*timewindow)

        with h5py.File(seizure_library_path, 'r+') as f:
            if annotation['dataset_name'] in f.keys():
                logging.info(str(annotation['dataset_name'])+' has more than one seizure!')
                #labels =  f[annotation['dataset_name']+'/labels']
                # todo wrap this in try, not possible to have end earlier than start of the seizure, will throw reverse selection error
                #labels[start_i:end_i] = 1
                #print('more than one seizure')

                #f[annotation['dataset_name']].attrs['chunked_annotation'] = np.vstack(
                #    [f[annotation['dataset_name']].attrs['chunked_annotation'], np.array([(120,160)]) ])

                #print(f[annotation['dataset_name']].attrs['chunked_annotation'])
                f[annotation['dataset_name']].attrs['precise_annotation'] = np.vstack(
                    [f[annotation['dataset_name']].attrs['precise_annotation'], np.array([(annotation['start'],annotation['end'])])])
                #print((f[annotation['dataset_name']].attrs['precise_annotation']))
                #print(dict(f[annotation['dataset_name']].attrs))

            else:
                group = f.create_group(annotation['dataset_name'])
                group.attrs['tid'] = annotation['tid']
                group.attrs['fs']  = fs
                group.attrs['scaled_and_filtered'] = scale_and_filter
                #group.attrs['chunked_annotation'] = np.array([(100,160)])
                group.attrs['precise_annotation'] = np.array([(annotation['start'],annotation['end'])])
                group.create_dataset('data', data = data_array, compression = "gzip", dtype='f4', chunks = data_array.shape)
                #labels[start_i:end_i] = 1 # indexing is fine, dont need to convert to array
                #group.create_dataset('labels', data = labels, compression = "gzip", dtype = 'i2', chunks = labels.shape)
            f.close()

    @staticmethod
    def _make_array_from_data(data, fs, timewindow):
        '''
        What ordering are we looking for here?
        '''
        n_traces = int(data.shape[0] / (fs * timewindow))
        dp_lost =  int(data.shape[0] % (fs * timewindow))
        if dp_lost > 0:
            data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, int(fs * timewindow)))
        else:
            data_array = np.reshape(data, newshape = (n_traces, int(fs * timewindow)))
        return data_array

    @staticmethod
    def fullpath_listdir(d):
        ''' returns full filepath,  excludes hidden files, starting with .'''
        return [os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')]

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
        files = [f for f in self.fullpath_listdir(ndf_dir) if f.endswith('.ndf')]

        # check ids
        ndf = NdfFile(files[0])
        if not tids == 'all':
            if not hasattr(tids, '__iter__'):
                tids = [tids]

        self.tids_for_parallel_conversion = tids
        print (str(len(files))+' Files for conversion. Transmitters: '+ str(self.tids_for_parallel_conversion))

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
        l = len(files)
        self.printProgress(0,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        for i, _ in enumerate(pool.imap(self.convert_ndf, files), 1):
            self.printProgress(i,l, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        pool.close()
        pool.join()

    def convert_ndf(self, filename):

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
                abs_savename = os.path.join(savedir, os.path.split(filename)[-1][:-4]+ndf_time+' tids_'+str(tids))
                ndf.save(save_file_name= abs_savename)
            else:
                logging.warning('Not all read tids:'+str(tids) +' were valid for '+str(os.path.split(filename)[1])+' skipping!')

        except Exception:
            print('Something unexpected went wrong loading '+str(tids)+' from '+mname+' :')
            #print('Valid ids are:'+str(ndf.tid_set))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print (traceback.print_exception(exc_type, exc_value,exc_traceback))
        return 0 # don't think i actually this
    # Print iterations progress

    def printProgress (self, iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : number of decimals in percent complete (Int)
            barLength   - Optional  : character length of bar (Int)
        """
        filledLength    = int(round(barLength * iteration / float(total)))
        percents        = round(100.00 * (iteration / float(total)), decimals)
        bar             = '█' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write('\n')
            sys.stdout.flush()
