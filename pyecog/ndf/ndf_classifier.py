
from __future__ import print_function
import pickle

import h5py
import numpy as np
from sklearn.metrics import classification_report

from pyecog.light_code.classifier import NetworkClassifer


class NDFClassifier():
    """
    Currently deprecated
    """

    def __init__(self):
        pass

    def load_annotated_h5(self, fpath, test_size = 0.25):
        with h5py.File(fpath, 'r') as f:
            fnames = f.keys()

            np.random.seed(7)
            fnames= np.random.permutation(fnames)

            n_test = int(len(fnames)*test_size)

            train_names = fnames[n_test:]
            test_names  = fnames[:n_test]
            self.train_features = np.vstack( f[name+'/features'] for name in train_names)
            self.train_labels   = np.vstack( f[name+'/labels'] for name in train_names)
            self.train_traces  = np.vstack( f[name+'/data'] for name in train_names)

            self.test_features = np.vstack( f[name+'/features'] for name in test_names)
            self.test_labels   = np.vstack( f[name+'/labels'] for name in test_names)
            self.test_traces   = np.vstack( f[name+'/data'] for name in test_names)

            print(self.test_features.shape)
            print(self.test_labels.shape)
            print(self.train_features.shape)
            print(self.train_labels.shape)

    def score_annotated_h5py(self, h5py_file):
        with h5py.File(h5py_file, 'r') as f:
            fnames = f.keys()
            features = np.vstack( f[name+'/features'] for name in fnames)
            targets  = np.vstack( f[name+'/labels'] for name in fnames)
        iX = self.classifier.imputer.transform(features)
        issX = self.classifier.std_scaler.transform(iX)

        predictions = self.classifier.r_forest.predict(issX)
        report = classification_report(targets, predictions)
        print(report)


    def train_clf(self):

        self.classifier = NetworkClassifer(self.train_features, self.train_labels, self.test_features, self.test_labels)
        # should throw in the params here!
        self.classifier.run()

        self.classifier.pca(n_components = 3)
        self.classifier.lda(n_components = 3, pca_reg = False, reg_dimensions = 9)
        self.classifier.lda_run()
        self.classifier.pca_run()

    def save_clf(self, savepath = None):
        if savepath is None:
            f = open('clf_dlog.p','wb')
        else:
            f = open(savepath, 'wb')
        pickle.dump(self.classifier, f)

    def load_clf(self, path = None):
        if path is None:
            self.classifier = pickle.load(open('clf_dlog.p','rb'))
        else:
            self.classifier = pickle.load(open(path,'rb'))



def main():
    clf_handler = NDFClassifier()
    dname = '/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.14/'
    clf_handler.load_annotated_h5(dname+'bundled_93.14_all')
    clf_handler.train_clf()
    clf_handler.save_clf()

    other_animal_annotations = '/Volumes/LaCie/Albert_ndfs/Data_03032016/Animal_93.8/bundled_annotations'

    clf_handler.load_clf('clf_dlog.p')
    print (clf_handler.classifier)
    clf_handler.score_annotated_h5py(other_animal_annotations)
if __name__ == "__main__":
    main()