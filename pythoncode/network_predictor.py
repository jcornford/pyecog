import pickle
import utils

import numpy as np
import pandas as pd

from network_loader import SeizureData
from extrator import FeatureExtractor
from make_pdfs import plot_traces

class Predictor():

    '''
    Todo:
    '''

    def __init__(self, clf_pickle_path=None, fs_dict_path='../pickled_fs_dictionary'):

        self.skipfiles = ('EX150515T11',
                        'EX180315T14',
                        'EX180515T4',
                        'EX200515T4.',)

        self.skip_dir = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/PV_ChR2/'

        if clf_pickle_path == None:
            clf_pickle_path = '../saved_clf'

        self.fs_dict  = pickle.load(open(fs_dict_path,'rb'))
        #for key in self.fs_dict:
            #print key, self.fs_dict[key]

        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
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
        if raw_load:
            self.dataobj = SeizureData(raw_path, fs_dict = self.fs_dict)
            self.dataobj.load_data()
            f = open('../'+savestring+'_saved','wb')
            pickle.dump(self.dataobj,f)

        else:
            assert saved_path != None
            self.dataobj = pickle.load(open(saved_path,'rb'))
        #print 'printing filename_list'
        #print self.dataobj.filename_list

        self.norm_data = utils.normalise(self.dataobj.data_array)
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
        self._string_fun2()
        self._write_to_excel()
        if make_pdfs:
            self.plot_pdfs()

    def plot_pdfs(self):
        plot_traces(self.norm_data,
                    self.preds,
                    savestring = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pdfs/'+self.savestring,
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

x = Predictor( clf_pickle_path = '../pickled_classifier')

makepdfs = True
x.assess_states(raw_path = None,
                savestring='PV_ARCH_predictions',
                raw_load = False,
                saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/PV_ARCH_pickled_tensec',
                make_pdfs= makepdfs)


x.assess_states(raw_path = None,
                savestring='PV_CHR2_predictions',
                raw_load = False,
                saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/PV_CHR2_pickled_tensec',
                make_pdfs= makepdfs)

x.assess_states(raw_path = None,
                savestring='SOM_CHR2_predictions',
                raw_load = False,
                saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/SOM_CHR2_pickled_tensec',
                make_pdfs= makepdfs)

