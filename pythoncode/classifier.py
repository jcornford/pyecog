'''

This is REALLY messy and needs attending too.. 20160310
Be able to pass in dictionary of params
be run score method outside of the training.


'''
from __future__ import print_function
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class NetworkClassifer():

    def __init__(self, features, labels, validation_features, validation_labels):
        self.features = features
        self.feature_labels = ['min','max','mean','skew','std','kurtosis','sum of absolute difference','baseline_n',
                               'baseline_diff','baseline_diff_skew','n_pks','n_vals','av_pk','av_val','av pk val range',
                               '1 hz','5 hz','10 hz','15 hz','20 hz','30 hz','60 hz','90 hz']

        self.labels = np.ravel(labels)
        self.validation_features = validation_features
        self.validation_labels = np.ravel(validation_labels)
        self.impute_and_scale()

    def impute_and_scale(self):
        print('Scaling and imputing training dataset...')
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(self.features)
        imputed_features = self.imputer.transform(self.features)
        self.std_scaler = StandardScaler()
        self.std_scaler.fit(imputed_features)
        self.iss_features = self.std_scaler.transform(imputed_features)
        print('Done')

        print('Scaling and imputing validation features using training dataset...')
        imputed_validation_features = self.imputer.transform(self.validation_features)
        self.iss_validation_features = self.std_scaler.transform(imputed_validation_features)
        print('Done')

    def _cross_validation(self,clf, k_folds = 5):
        self.scores = cross_validation.cross_val_score(clf, self.iss_features, self.labels, cv=k_folds,n_jobs=5, scoring = 'roc_auc')

    def randomforest_info(self, max_trees = 1000, step = 40, k_folds = 5):
        print('Characterising R_forest. Looping through trees: ')
        self.treedata = np.zeros((max_trees/step, 10))
        for i,n_trees in enumerate(np.arange(0, max_trees,step)):
            if n_trees == 0:
                n_trees = 1

            r_forest = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0, )
            scores = cross_validation.cross_val_score(r_forest, self.iss_features, self.labels, cv=k_folds,n_jobs=5)
            r_forest_full = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest_full.fit(self.iss_features,self.labels)
            self.treedata[i,0] = n_trees
            self.treedata[i,1] = scores.mean()
            self.treedata[i,2] = scores.std()
            # now add the test dataset - score
            self.treedata[i,3] = r_forest_full.score(self.iss_validation_features, self.validation_labels)

            r_forest_lda = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest_lda_full = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest_lda_full.fit(self.lda_iss_features,self.labels)
            lda_scores = cross_validation.cross_val_score(r_forest_lda, self.lda_iss_features, self.labels, cv=k_folds,n_jobs=5)
            self.treedata[i,4] = lda_scores.mean()
            self.treedata[i,5] = lda_scores.std()
            self.treedata[i,6] = r_forest_lda_full.score(self.lda_iss_validation_features, self.validation_labels)


            r_forest_pca = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest_pca_full = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest_pca_full.fit(self.pca_iss_features,self.labels)
            pca_scores = cross_validation.cross_val_score(r_forest_pca, self.pca_iss_features, self.labels, cv=k_folds,n_jobs=5)
            self.treedata[i,7] = pca_scores.mean()
            self.treedata[i,8] = pca_scores.std()
            self.treedata[i,9] = r_forest_pca_full.score(self.pca_iss_validation_features, self.validation_labels)

    def pca(self,n_components = 6):
        self.pca = PCA(n_components)
        self.pca_iss_features = self.pca.fit_transform(self.iss_features)
        self.pca_iss_validation_features = self.pca.transform(self.iss_validation_features)

    def lda(self,n_components = 2, pca_reg = True, reg_dimensions = 10):
        self.lda = LinearDiscriminantAnalysis(n_components = n_components, solver='eigen', shrinkage='auto')
        #self.lda = LDA(n_components)
        if pca_reg:
            self.pca_reg = PCA(reg_dimensions)
            pca_reg_features = self.pca_reg.fit_transform(self.iss_features)
            self.lda_iss_features = self.lda.fit_transform(pca_reg_features,self.labels)
            pca_reg_validation_features = self.pca_reg.transform(self.iss_validation_features)
            self.lda_iss_validation_features = self.lda.transform(pca_reg_validation_features)
        else:
            self.lda_iss_features = self.lda.fit_transform(self.iss_features,self.labels)
            self.lda_iss_validation_features = self.lda.transform(self.iss_validation_features)

    def lda_run(self, k_folds = 5):
        self.r_forest_lda = RandomForestClassifier(n_estimators=2000,
                                                   n_jobs=5,
                                                   max_depth=None,
                                                   min_samples_split=2,
                                                   random_state =7,
                                                   max_leaf_nodes=None,
                                                   min_samples_leaf=2,
                                                   criterion='gini',
                                                   max_features='sqrt',
                                                   class_weight='balanced')

        self.lda_scores = cross_validation.cross_val_score(self.r_forest_lda,
                                                           self.lda_iss_features,
                                                           self.labels,
                                                           cv=k_folds,
                                                           n_jobs=5)
        print("Cross validation Random Forest performance LDA: Accuracy: %0.2f (std %0.2f)" % (self.lda_scores.mean()*100, self.lda_scores.std()*100))
        self.r_forest_lda.fit(self.lda_iss_features,self.labels)
        print(str(self.r_forest_lda.score(self.lda_iss_validation_features,
                                      self.validation_labels)*100)+ 'LDA test-set performance')

        y_true = self.validation_labels
        y_pred = self.r_forest_lda.predict(self.lda_iss_validation_features)
        target_names = ['S1','S2','S3','S4']
        report = classification_report(y_true, y_pred, target_names=target_names)
        print('Random forest report lda')
        print(report)

        ##### Hacky way to export features, so can optimise RF etc ######
        train_X = pd.DataFrame(self.lda_iss_features)
        train_y  = pd.DataFrame(self.labels)
        training = pd.concat([train_X,train_y], axis = 1)
        training.to_csv('/Users/Jonathan/Dropbox/Data_sharing_VMJC/training_lda.csv', index = False)

        test_X = pd.DataFrame(self.lda_iss_validation_features)
        test_y = pd.DataFrame(self.validation_labels)
        test = pd.concat([test_X,test_y], axis = 1)
        test.to_csv('/Users/Jonathan/Dropbox/Data_sharing_VMJC/test_lda.csv', index = False)

        train_X = pd.DataFrame(self.iss_features)
        train_y  = pd.DataFrame(self.labels)
        training = pd.concat([train_X,train_y], axis = 1)
        training.to_csv('/Users/Jonathan/Dropbox/Data_sharing_VMJC/training.csv', index = False)

        test_X = pd.DataFrame(self.iss_validation_features)
        test_y = pd.DataFrame(self.validation_labels)
        test = pd.concat([test_X,test_y], axis = 1)
        test.to_csv('/Users/Jonathan/Dropbox/Data_sharing_VMJC/test.csv', index = False)

    def pca_run(self,k_folds = 5):
        self.r_forest_pca = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0)
        self.pca_scores = cross_validation.cross_val_score(self.r_forest_pca, self.pca_iss_features, self.labels, cv=k_folds,n_jobs=5)
        print("Cross validation RF performance PCA: Accuracy: %0.2f (std %0.2f)" % (self.pca_scores.mean()*100, self.pca_scores.std()*100))

        self.r_forest_pca.fit(self.pca_iss_features,self.labels)
        print(str(self.r_forest_pca.score(self.pca_iss_validation_features, self.validation_labels))+ 'PCA test-set performance ')

    def run(self):

        r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0, class_weight='balanced')
        self._cross_validation(r_forest)
        print("Cross validation RF performance: Accuracy: %0.2f (std %0.2f)" % (self.scores.mean()*100, self.scores.std()*100))

        self.r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0, class_weight='balanced')
        self.r_forest.fit(self.iss_features,self.labels)

        print(str(self.r_forest.score(self.iss_validation_features, self.validation_labels))+ 'randomforest test-set performance')

        y_true = self.validation_labels
        y_pred = self.r_forest.predict(self.iss_validation_features)
        target_names = ['inter-ictal', 'ictal']
        target_names = ['S1','S2','S3','S4']
        t = classification_report(y_true, y_pred, target_names=target_names)
        print('Random forest report:')
        print(t)


        return None

