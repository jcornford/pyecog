import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyecog.ndf.ndfconverter import NdfFile


class Explorer():
    '''
    Just a mini version of ndf main to more easily look at sections of data.
    '''

    def __init__(self, normalise_seconds = True):
        self.dir_path = '/Users/Jonathan/Dropbox/EEG/'
        self.annotation_file = 'seizures_Rat8_20-271015.xlsx'

        self.normalise_seconds = True


        annotations = pd.read_excel(self.dir_path+self.annotation_file, index_col=False)

        annotations.columns = ['fname','start','end']
        #self.annotations = annotations.dropna(axis=0, how='all')

        fname = annotations.iloc[1,0]
        start = annotations.iloc[1,1]
        end = annotations.iloc[1,2]
        duration = start - end

        print start,end,duration,file
        fname = 'M1445362612.ndf'
        ndf = NdfFile(self.dir_path+'ndf/'+fname, print_meta=True)
        ndf.load(8)

        np.set_printoptions(precision=3, suppress = True)
        ictal_time = ndf.time[(ndf.time >= start) & (ndf.time <= end)]
        ictal_data = ndf.data[(ndf.time >= start) & (ndf.time <= end)]

        plt.plot(ictal_time,ictal_data)
        #plt.plot(ndf.time[:5120], ndf.data[:5120])
        plt.show()

if __name__ == '__main__':
    n = Explorer()
