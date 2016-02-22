import pickle
import pythoncode.utils as utils

import numpy as np
import pandas as pd

from pythoncode.network_loader import SeizureData
from pythoncode.extrator import FeatureExtractor
from pythoncode.make_pdfs import plot_traces


import matplotlib.pyplot as plt
import h5py



class Predictor():

    '''
    Todo:
    '''

    def __init__(self, clf_pickle_path, fs_dict_path='../pickled_fs_dictionary'):

        #self.fs_dict  = pickle.load(open(fs_dict_path,'rb'))
        #for key in self.fs_dict:
            #print key, self.fs_dict[key]
        pickle.load(open('/Volumes/LACIE SHARE/pickled_classifier','rb'))
        #self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.r_forest = self.classifier.r_forest
        self.r_forest_lda = self.classifier.r_forest_lda
        self.lda = self.classifier.lda

        print self.lda
        #print self.r_forest_lda


    def assess_states(self, raw_path = None, downsample_rate = None, savestring = 'example',
                      threshold = 65,
                      raw_load = True,
                      saved_path = None,
                      make_pdfs = True):

        self.threshold = '65' # 'sureity' threshold
        self.savestring = savestring


        with h5py.File('M1455096626.hdf5', 'r') as hf:

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

            index = data.shape[0]/5120
            print index, 'is divded by 5120'

            data_array = np.reshape(data[:5120*index], (index,5120,))
            print data_array.shape
            #plt.figure(figsize = (20,10))
            #plt.plot(data_array[40,:])

            #plt.show()
        self.data_array = data_array



        self.norm_data = utils.normalise(self.data_array)
        feature_obj = FeatureExtractor(self.norm_data)

        i_features = self.classifier.imputer.transform(feature_obj.feature_array)
        iss_features = self.classifier.std_scaler.transform(i_features)
        lda_iss_features = self.lda.transform(iss_features)

        np.set_printoptions(precision=3, suppress = True)

        #self.pred_table = self.r_forest.predict_proba(iss_features)*100
        #self.preds = self.r_forest.predict(iss_features)

        self.pred_table = self.r_forest_lda.predict_proba(lda_iss_features)*100
        self.preds = self.r_forest_lda.predict(lda_iss_features)

        self.predslist = list(self.preds) # why need this?
        self.predslist[self.predslist == 4] = 'Baseline'
        self.max_preds = np.max(self.pred_table, axis = 1)
        #print pred_table
        self.threshold_for_mixed = np.where(self.max_preds < int(self.threshold),1,0) # 1 when below
        #self._string_fun2()
        #self._write_to_excel()
        if make_pdfs:
            self.plot_pdfs()

    def plot_pdfs(self):
        plot_traces(self.norm_data,
                    self.preds,
                    savestring = '/Volumes/LACIE SHARE/Andys ndf/pdfs/'+self.savestring,
                    prob_thresholds= self.threshold_for_mixed)

    def _string_fun2(self):
        '''
        This method is for the full data, vm gave in 2016/01
        '''
        self.nameframe =  pd.DataFrame(columns = ['Date', 'ID', 'File Start', 'File End', 'Pulse Time'])
        for i,f in enumerate(self.dataobj.filename_list):

            f =  f.split('/')[-1]
            try:
                date = f.split('X')[1].split('T')[0]
            except IndexError:
                date = f.split('x')[1].split('T')[0]
            t_start = '0'
            t_end   = f.split('_')[-1]
            t_onset = f.split('_')[1]
            transmitter = f.split('_')[0].split(date)[-1]
            #print [date, transmitter, t_start, t_end, t_onset]
            self.nameframe.loc[i] = [date, transmitter, t_start, t_end, t_onset]
            #print f

    def _string_fun(self):
        self.nameframe =  pd.DataFrame(columns = ['Date', 'ID', 'File Start', 'File End', 'Pulse Time'])

        for i,f in enumerate(self.dataobj.filename_list):
            #print f

            f =  f.split('/')[-1]
            s_brackets = f.split('[', 1)[1].split(']')[0]
            s_brackets_start = s_brackets.split('-')[0]
            t_end = s_brackets.split('-')[-1].split('s')[0]
            try:
                t_start = int(s_brackets_start)
            except:
                t_start = int(s_brackets_start.split(' ')[-1])
            date = f.split('X')[1].split('t')[0].strip('-')
            transmitter = f.split('_')[0].split('r')[-1]
            t_onset = float(f.split(']')[-1]) + t_start

            self.nameframe.loc[i] = [date, transmitter, t_start, t_end, t_onset]
        #print self.nameframe.head()

    def _write_to_excel(self):
        sheet = pd.DataFrame(self.pred_table)
        pred = pd.DataFrame(self.predslist,columns=['Index'])
        max_preds = pd.DataFrame(self.max_preds)
        max_preds.columns = ['Max']
        sheet.columns = ['State1','State2','State3','Baseline']
        frames = [self.nameframe, sheet, max_preds, pred]
        vmsheet = pd.concat(frames,axis = 1)
        print vmsheet.head()
        writer = pd.ExcelWriter('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/'+self.savestring+'.xlsx',engine = 'xlsxwriter')
        vmsheet.to_excel(writer,index = True,sheet_name = 'Pulse prediction' )
        workbook = writer.book
        worksheet = writer.sheets['Pulse prediction']
        percent_fmt = workbook.add_format({'num_format': '0.00', 'bold': False})
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        worksheet.set_column('G:J',12,percent_fmt)
        color_range = "K2:K{}".format(len(self.dataobj.filename_list)+1)
        #worksheet.conditional_format(color_range, {'type': 'top',
        #                                   'value': '20',
        #                                   'format': format1})
        worksheet.conditional_format(color_range, {'type': 'cell',
                                                   'criteria': '<=',
                                           'value': self.threshold,
                                           'format': format1})
        writer.save()

x = Predictor( clf_pickle_path = '/Users/jonathan/PycharmProjects/networkclassifer/pickled_classifier')

makepdfs = True

x.assess_states(raw_path = '/Volumes/LACIE SHARE/Andys ndf/',
                savestring='ndf_predictions',
                raw_load = False,
                saved_path = '/Volumes/LACIE SHARE/Andys ndf',
                make_pdfs= makepdfs)


