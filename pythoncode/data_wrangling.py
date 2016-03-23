
import h5py
import os

import numpy as np

from converter import NDFLoader
from make_pdfs import plot_traces_hdf5, plot_traces

class DataHandler():
    '''
    Class to handle all ingesting of data and outputing it in a format for the Classifier Handler

    TODO:
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
                        print('plotting transmitter id ' + str(tid))
                        data = np.array(datadict[tid]['data'])

                        n_traces = data.shape[0] / (fs * timewindow)
                        print('There are '+ str(n_traces) + ' traces')
                        dp_lost =  data.shape[0] % (fs * timewindow)

                        data_array = np.reshape(data[:-dp_lost], newshape = (n_traces, (fs * timewindow)))
                        data_array = self._normalise(data_array)

                        figure_folder = os.path.join(savedir, path.split('/')[-1].split('.')[0]+'_id'+str(tid))
                        print figure_folder
                        if not os.path.exists(figure_folder):
                            os.makedirs(figure_folder)

                        black = np.ones((data.shape[0]))
                        plot_traces_hdf5(data_array,
                                    labels = black,
                                    savepath = figure_folder,
                                    filename = path.split('/')[-1].split('.')[0],
                                    format_string = format,
                                    trace_len_sec = timewindow)


    def append_to_annotated_dataset(self):
        pass


    def make_annotated_dataset(self):

        '''
        This needs to be done...!

        '''

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

    # Here kind of endss the nonesense.   # #######
    #######
            ######




def main():
    handler = DataHandler()
    #handler.convert_ndf('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.8/NDF/', ids = 14)
    handler.make_figures_for_labeling_ndfs('/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.8/converted_ndfs/M1454346216.hdf5',
                                           timewindow=5)
if __name__ == "__main__":
    main()