import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn import cross_validation


class NetworkClassifer():

    def __init__(self, features, labels, validation_features, validation_labels):
        self.features = features
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
        scores = cross_validation.cross_val_score(clf, self.iss_features, self.labels, cv=k_folds,n_jobs=5)
        return scores
        #print 'Cross validation performance',scores
        #print("Accuracy: %0.2f (std %0.2f)" % (scores.mean()*100, scores.std()*100))

    def randomforest_info(self, max_trees = 2000, step = 10, kfolds = 5):
        self.treedata = np.zeros((max_trees/step,4))
        for i,n_trees in enumerate(np.arange(0,max_trees,step)):
            if n_trees == 0:
                n_trees = 1
            #print i, n_trees
            r_forest = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)

            self.treedata[i,0] = n_trees
            self.treedata[i,1] = r_forest.score(self.X_train,self.y_train)
            self.treedata[i,2] = r_forest.score(self.X_test,self.y_test)
            self.treedata[i,3] = r_forest.score(self.iss_validation_features, self.validation_labels)
        np.savetxt('../treedata.csv',self.treedata, delimiter=',')

    def _pca(self,n_components = 6):
        self.pca = PCA(n_components)
        try:
            self.pca_iss_features = self.fit_transform(self.iss_features)
        except:
            print 'Error in PCA call: Have you scaled your features yet?'

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

    def run(self):

        #######################################
        #r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
        self.X_train,self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.iss_features, self.labels, test_size=0.5, random_state=3)
        #r_forest.fit(self.X_train,self.y_train)
        ########################################

        r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state =0)
        self._cross_validation(r_forest)

        r_forest = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
        r_forest.fit(self.iss_features,self.labels)

        #print r_forest.score(self.X_train,self.y_train), 'randomforest train performance'
        #print r_forest.score(self.X_test,self.y_test), 'randomforest test performance'
        print r_forest.score(self.iss_validation_features, self.validation_labels), 'randomforest valdiation set performance \n'

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
