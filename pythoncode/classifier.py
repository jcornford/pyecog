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
        self.features0616 = features
        self.labels0616 = labels

        self.validation_features = validation_features
        self.validation_labels = validation_labels

    def randomforest_info(self, max_trees = 2000, step = 10):
        self.treedata = np.zeros((max_trees,3))
        for i in np.arange(0,max_trees,step):
            n = i + 1
            r_forest = RandomForestClassifier(n_estimators=n,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
            r_forest.fit(self.X_train,self.y_train)
            self.treedata[i,0] = r_forest.score(self.X_train,self.y_train)
            self.treedata[i,1] = r_forest.score(self.X_test,self.y_test)
            self.treedata[i,2] = r_forest.score(self.vfeatsiss,self.vlabels)
        np.savetxt('../treedata.csv',self.treedata, delimiter=',')

    def run(self):
        self.imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.imputer.fit(self.features0616)
        features0616i = self.imputer.transform(self.features0616)
        self.std_scaler = preprocessing.StandardScaler().fit(features0616i)

        self.features0616iss = self.std_scaler.transform(features0616i)
        self.labels0616_flat = np.ravel(self.labels0616)

        pca0616_6d = PCA(n_components = 6)
        features0616iss_6d = pca0616_6d.fit_transform(self.features0616iss)

        vfeatsi = self.imputer.transform(self.validation_features)
        self.vfeatsiss = self.std_scaler.transform(vfeatsi)
        self.vlabels = np.ravel(self.validation_labels)

        #########################################
        self.X_train,self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.features0616iss, self.labels0616_flat, test_size=0.5, random_state=3)
        r_forest16 = RandomForestClassifier(n_estimators=2000,n_jobs=5, max_depth=None, min_samples_split=1, random_state=0)
        r_forest16.fit(self.X_train,self.y_train)

        print r_forest16.score(self.X_train,self.y_train), 'randomforest train performance'
        print r_forest16.score(self.X_test,self.y_test), 'randomforest test performance'
        print r_forest16.score(self.vfeatsiss,self.vlabels), 'randomforest valdiation set performance \n'

        svm_clf16 = SVC()
        svm_clf16.fit(self.X_train,self.y_train)
        print svm_clf16.score(self.X_train,self.y_train), 'SVC rbf train performance'
        print svm_clf16.score(self.X_test,self.y_test), 'SVC rbf test performance'
        print svm_clf16.score(self.vfeatsiss,self.vlabels), 'SVC rbf valdiation set performance \n'

        n_neighbors = 4
        clf = KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(self.X_train, self.y_train)
        print clf.score(self.X_train,self.y_train), 'KNN train performance'
        print clf.score(self.X_test,self.y_test), 'KNN test performance'
        print clf.score(self.vfeatsiss,self.vlabels), 'KNN valdiation set performance \n'

        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train)
        print clf.score(self.X_train,self.y_train), 'D-tree test performance'
        print clf.score(self.X_test,self.y_test), 'D-tree test performance'
        print clf.score(self.vfeatsiss,self.vlabels), 'D-tree valdiation set performance \n'
