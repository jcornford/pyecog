from __future__ import print_function
import pickle

import numpy as np
import pandas as pd

import utils
from pyecog.light_code.network_loader import SeizureData
from pyecog.light_code.extractor import FeatureExtractor
from pyecog.light_code.make_pdfs import plot_traces


class Predictor():

    '''
    Class to predict state from unlabeled data.

    Data should be in th same format as the training data - stacked into traces and saved as a hdf5 file.
    '''

    def __init__(self, clf_pickle_path=None, threshold = 50):

        '''

        This currently (messily) using a fs_dict, for sampling frequency of files.
        And a tuple of skipfiles
        Args:
            clf_pickle_path: the pickled classifier to use
            fs_dict_path: dictionary made by abf_loader

        Returns:

        '''


        self.sg_filter_window_size = 7
        self.sg_filter_window_order = 3
        self.threshold = threshold # 'sureity' threshold

        self.classifier = pickle.load(open(clf_pickle_path,'rb'))
        self.lda = self.classifier.lda
        self.r_forest_lda = self.classifier.r_forest_lda


    def load_traces_to_classify(self,path, fs_dict_path='../pickled_fs_dictionary',):

        self.skipfiles = ('EX150515T11',
                        'EX180315T14',
                        'EX180515T4',
                        'EX200515T4.',)

        self.skip_dir = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/PV_ChR2/'


        self.fs_dict  = pickle.load(open(fs_dict_path,'rb'))

        # N.B only predicting with the lda r_forest at the moment
        #self.r_forest = self.classifier.r_forest

        self.dataobj = SeizureData(path, fs_dict = self.fs_dict)
        self.dataobj.load_data()
        f = open('../'+savestring+'_saved','wb')
        pickle.dump(self.dataobj,f)

    def assess_states(self, saved_path,
                      savestring = 'example',
                      pdf_savepath = '../',
                      make_pdfs = True):

        self.savestring = savestring
        self.pdf_savepath = pdf_savepath


        self.dataobj = pickle.load(open(saved_path,'rb'))

        self.norm_data = utils.normalise(self.dataobj.data_array)
        self.norm_data = utils.filterArray(self.norm_data,
                                           window_size= self.sg_filter_window_size,
                                           order=self.sg_filter_window_order)

        feature_obj = FeatureExtractor(self.norm_data)

        i_features = self.classifier.imputer.transform(feature_obj.feature_array)
        iss_features = self.classifier.std_scaler.transform(i_features)
        lda_iss_features = self.lda.transform(iss_features)

        # predict probability and also the actual state
        self.pred_table = self.r_forest_lda.predict_proba(lda_iss_features)*100
        self.preds = self.r_forest_lda.predict(lda_iss_features)

        # Make stuff for the excel sheet
        self.predslist = list(self.preds) # why need this?
        self.predslist[self.predslist == 4] = 'Baseline'
        self.max_preds = np.max(self.pred_table, axis = 1)
        self.threshold_for_mixed = np.where(self.max_preds < int(self.threshold),1,0) # 1 when below

        # do the 1st vs 2nd most likely states
        self.sorted_pred = np.sort(self.pred_table, axis = 1)
        self.ratio = np.divide(self.sorted_pred[:,2],self.sorted_pred[:,3])
        self.threshold_for_ratio = np.where(self.ratio > 0.5,1,0) # 1 when below

        # combine the two measures
        self.combined_pass = np.logical_or(self.threshold_for_mixed,self.threshold_for_ratio)

        self._string_fun2()
        self._write_to_excel()
        if make_pdfs:
            plot_traces(self.norm_data,
                        self.preds,
                        savepath= self.pdf_savepath + self.savestring,
                        #savestring = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pdfs0302/'+self.savestring,
                        prob_thresholds= self.combined_pass)

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
        '''
        This is for the original style of file input...
        '''
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
        sheet.columns = ['State1','State2','State3','Baseline']

        pred = pd.DataFrame(self.predslist,columns=['Index'])

        max_preds = pd.DataFrame(self.max_preds)
        max_preds.columns = ['Max']

        master_exc =  pd.DataFrame(self.combined_pass,columns=['Exclude'])

        ratio = pd.DataFrame(self.ratio, columns = ['2v1_Ratio'])

        frames = [self.nameframe, sheet, max_preds, ratio, master_exc, pred]
        vmsheet = pd.concat(frames,axis = 1)
        #print vmsheet.head()
        writer = pd.ExcelWriter('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/'+self.savestring+'.xlsx',engine = 'xlsxwriter')
        vmsheet.to_excel(writer,index = True,sheet_name = 'Pulse prediction' )
        workbook = writer.book
        worksheet = writer.sheets['Pulse prediction']
        percent_fmt = workbook.add_format({'num_format': '0.00', 'bold': False})
        format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
        worksheet.set_column('G:J',12,percent_fmt)

        color_range = "K2:K{}".format(len(self.dataobj.filename_list)+1)
        worksheet.conditional_format(color_range, {'type': 'cell',
                                                   'criteria': '<=',
                                           'value': self.threshold,
                                           'format': format1})

        ratio_range = "L2:L{}".format(len(self.dataobj.filename_list)+1)
        worksheet.conditional_format(ratio_range, {'type': 'cell',
                                                   'criteria': '>=',
                                           'value': 0.5,
                                           'format': format1})
        ratio_range = "M2:M{}".format(len(self.dataobj.filename_list)+1)
        worksheet.conditional_format(ratio_range, {'type': 'cell',
                                                   'criteria': '=',
                                           'value': 1,
                                           'format': format1})
        writer.save()

def main():
    # there is also a pickled classifier in the directory just above here
    x = Predictor( clf_pickle_path = '/Volumes/LACIE SHARE/VM_data/classifier_hdf5data/pickled_classifier_20160223',
                   threshold= 50)


    makepdfs = True
    x.assess_states(savestring='PV_ARCH_predictions_20160310',
                    pdf_savepath='/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pdfs0310/',
                    saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/PV_ARCH_pickled_tensec',
                    make_pdfs= makepdfs)

    '''
    raw_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/PV_ARCH/ConvertedFiles/',
    x.assess_states(raw_path = None,
                    savestring='PV_CHR2_predictions_20160310',
                    raw_load = False,
                    pdf_savepath='/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pdfs0310/',
                    saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/PV_CHR2_pickled_tensec',
                    make_pdfs= makepdfs)

    x.assess_states(raw_path = None,
                    savestring='SOM_CHR2_predictions_20160310',
                    raw_load = False,
                    pdf_savepath='/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pdfs0310/',
                    saved_path = '/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/pickled_tensec_dobj/SOM_CHR2_pickled_tensec',
                    make_pdfs= makepdfs)
    '''
if __name__ == "__main__":
    main()





    '''
    ###################################
    # adding to the training dataset!
    for_training = True
    certified_training = []
    if for_training:
        #print 'here'
        t_labels = pd.read_excel('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/labelling_for_training/090216_PVArch_test0_7.xlsx').values
        #print t_labels
        for i in range(t_labels.shape[0]):
            if t_labels[i,1] != 4: # in this excel doc we usde 4 for mixed state! :/ and 0 for baseline
                certified_training.append(i)


        #print t_labels[certified_training,:]
        f = open('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/labelling_for_training/new_training_data_2016_02_09','wb')
        pickle.dump(self.dataobj.data_array[certified_training,:],f)
        print self.dataobj.data_array[certified_training,:].shape

        t_labels_vet = t_labels[certified_training,1]
        t_labels_vet[t_labels_vet==0] = 4 # need to correct to use the baseline index 4, as normal
        f = open('/Volumes/LACIE SHARE/VM_data/All_Data_Jan_2016/labelling_for_training/new_training_labels_2016_02_09','wb')
        pickle.dump(t_labels_vet,f)
        print t_labels_vet.shape
        print t_labels_vet

    exit()

    ###################################
    '''