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

    def __init__(self, clf_pickle_path = None, ):

        if clf_pickle_path == None:
            clf_pickle_path = '../saved_clf'
        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.r_forest = self.classifier.r_forest
        self.r_forest_lda = self.classifier.r_forest_lda
        #print self.r_forest
        #print self.r_forest_lda


    def assess_states(self,fpath, downsample_rate = None, savestring = 'example', threshold = 65):
        self.threshold = '65'
        self.savestring = savestring

        self.dataobj = SeizureData(fpath, amount_to_downsample = downsample_rate)
        self.dataobj.load_data()
        #print 'printing filename_list'
        #print self.dataobj.filename_list

        self.norm_data = utils.normalise(self.dataobj.data_array)
        feature_obj = FeatureExtractor(self.norm_data)
        i_features = self.classifier.imputer.transform(feature_obj.feature_array)
        iss_features = self.classifier.std_scaler.transform(i_features)
        np.set_printoptions(precision=3, suppress = True)
        self.pred_table = self.r_forest.predict_proba(iss_features)*100
        self.preds = self.r_forest.predict(iss_features)
        self.predslist = list(self.preds)
        self.predslist[self.predslist == 4] = 'Baseline'
        self.max_preds = np.max(self.pred_table, axis = 1)
        #self._string_fun()
        #self._write_to_excel()

    def plot_traces(self):
        plot_traces(self.norm_data, self.preds, savestring = self.savestring)

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
        writer = pd.ExcelWriter('../'+self.savestring+'.xlsx',engine = 'xlsxwriter')
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

'''
nc = Predictor( clf_pickle_path = '../pickled_classifier')
nc.assess_states('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/PV_Arch_test/ConvertedFiles/',
                 downsample_rate = 20)
'''
x = Predictor( clf_pickle_path = '../pickled_classifier')
x.assess_states('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/PV_Arch_test/ConvertedFiles/',downsample_rate = 20)


#x.assess_states('/Users/Jonathan/PhD/Seizure_related/batchSept_UC_40/c1',downsample_rate=20, savestring = '2015_12_09_predictions_40')
'''
x.plot_traces()

x20 = Predictor()
x20.assess_states('/Users/Jonathan/PhD/Seizure_related/batchSept_UC_20',downsample_rate=40, savestring = '2015_12_09_predictions_20')
x20.plot_traces()
'''