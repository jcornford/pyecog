import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
#from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn import cross_validation
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

        self._impute_and_scale()

    def _impute_and_scale(self):
        print 'Scaling and imputing training dataset...',
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(self.features)
        imputed_features = self.imputer.transform(self.features)

        self.std_scaler = preprocessing.StandardScaler()
        self.std_scaler.fit(imputed_features)
        self.iss_features = self.std_scaler.transform(imputed_features)
        print 'Done'
        print 'Scaling and imputing validation features using training dataset...',
        imputed_validation_features = self.imputer.transform(self.validation_features)
        self.iss_validation_features = self.std_scaler.transform(imputed_validation_features)
        print 'Done'

    def _cross_validation(self,clf, k_folds = 5):
        self.scores = cross_validation.cross_val_score(clf, self.iss_features, self.labels, cv=k_folds,n_jobs=5)

    def randomforest_info(self, max_trees = 1000, step = 40, k_folds = 5):
        print 'Characterising R_forest. Looping through trees: ',
        self.treedata = np.zeros((max_trees/step, 10))
        for i,n_trees in enumerate(np.arange(0, max_trees,step)):
            if n_trees == 0:
                n_trees = 1
            print n_trees,
            r_forest = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
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
            print self.treedata[i,6]

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

    def knn_info(self, kmax=100 ):
        self.knndata = np.zeros((kmax,4))
        for i in range(kmax):
            k = i+1
            knn = KNeighborsClassifier(k, weights='distance')
            knn.fit(self.X_train, self.y_train)
            self.knndata[i,0] = k
            self.knndata[i,1] = knn.score(self.X_train,self.y_train)
            self.knndata[i,2] = knn.score(self.X_test,self.y_test)
            self.knndata[i,3] = knn.score(self.iss_validation_features, self.validation_labels)
        np.savetxt('../knndata.csv',self.knndata, delimiter=',')

    def lda_run(self, k_folds = 5):
        self.r_forest_lda = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0)
        self.lda_scores = cross_validation.cross_val_score(self.r_forest_lda, self.lda_iss_features, self.labels, cv=k_folds,n_jobs=5)
        print("Cross validation RF performance LDA: Accuracy: %0.2f (std %0.2f)" % (self.lda_scores.mean()*100, self.lda_scores.std()*100))

        self.r_forest_lda.fit(self.lda_iss_features,self.labels)
        print self.r_forest_lda.score(self.lda_iss_validation_features, self.validation_labels), 'LDA test-set performance \n'

    def pca_run(self,k_folds = 5):
        self.r_forest_pca = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0)
        self.pca_scores = cross_validation.cross_val_score(self.r_forest_pca, self.pca_iss_features, self.labels, cv=k_folds,n_jobs=5)
        print("Cross validation RF performance PCA: Accuracy: %0.2f (std %0.2f)" % (self.pca_scores.mean()*100, self.pca_scores.std()*100))

        self.r_forest_pca.fit(self.pca_iss_features,self.labels)
        print self.r_forest_pca.score(self.pca_iss_validation_features, self.validation_labels), 'PCA test-set performance \n'

    def run(self):

        #######################################
        #r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
        self.X_train,self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.iss_features, self.labels, test_size=0.5, random_state=3)
        #r_forest.fit(self.X_train,self.y_train)
        ########################################

        r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0)
        self._cross_validation(r_forest)
        print("Cross validation RF performance: Accuracy: %0.2f (std %0.2f)" % (self.scores.mean()*100, self.scores.std()*100))

        self.r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
        self.r_forest.fit(self.iss_features,self.labels)

        #print r_forest.score(self.X_train,self.y_train), 'randomforest train performance'
        #print r_forest.score(self.X_test,self.y_test), 'randomforest test performance'

        print self.r_forest.score(self.iss_validation_features, self.validation_labels), 'randomforest test-set performance \n'

        svm_clf = SVC()
        svm_clf.fit(self.X_train,self.y_train)
        print svm_clf.score(self.X_train,self.y_train), 'SVC rbf train performance'
        print svm_clf.score(self.X_test,self.y_test), 'SVC rbf test performance'
        print svm_clf.score(self.iss_validation_features, self.validation_labels), 'SVC rbf valdiation set performance \n'

        n_neighbors = 4
        knn = KNeighborsClassifier(n_neighbors, weights='distance')
        knn.fit(self.X_train, self.y_train)
        print knn.score(self.X_train,self.y_train), 'KNN train performance'
        print knn.score(self.X_test,self.y_test), 'KNN test performance'
        print knn.score(self.iss_validation_features, self.validation_labels), 'KNN valdiation set performance \n'

        dtc = DecisionTreeClassifier()
        dtc.fit(self.X_train, self.y_train)
        print dtc.score(self.X_train,self.y_train), 'D-tree test performance'
        print dtc.score(self.X_test,self.y_test), 'D-tree test performance'
        print dtc.score(self.iss_validation_features, self.validation_labels), 'D-tree valdiation set performance \n'
