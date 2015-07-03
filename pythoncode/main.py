import numpy as np
import matplotlib.pyplot as plt

# Basic script to run network state classfication

from loadSeizureData  import LoadSeizureData
from classifierTester import ClassifierTester
from relabeling_functions import relabel, reorder
from visualisation import plots

from extractors.basicFeatures    import BasicFeatures
from extractors.freqfeatures     import FreqFeatures
from extractors.waveletfeatures import WaveletFeatures
from extractors.postcrossingfreqfeatures import PostCrossingPower

from classifiers.randomForestClassifier import RandomForest
from classifiers.svm import SupportVecClf
from classifiers.neighbors import KNeighbors

dirpath = '/Users/Jonathan/Documents/PhD/Seizure_related/Classified'
dataobj = LoadSeizureData(dirpath)
dataobj.load_data()
dataobj = relabel(dataobj)
dataobj = reorder(dataobj)

basicStatsExtractor = BasicFeatures()
wavelets = WaveletFeatures()
pcp = PostCrossingPower()
fourier = FreqFeatures()

dataobj.extract_feature_array([basicStatsExtractor,pcp,wavelets])
print dataobj.features.shape
#dataobj.extract_feature_array([wavelets, basicStatsExtractor])

# NOW GENERATE TRAINING AND TEST

# TRAINING FEATURE MATRIX MODIFICATION AND NORMALISATION
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(dataobj.features)
dataobj.features = std_scale.transform(dataobj.features)
minmax_scale = preprocessing.MinMaxScaler().fit(dataobj.features)
#dataobj.features = minmax_scale.transform(dataobj.features)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(dataobj.features)
dataobj.lda = pca.transform(dataobj.features)
#from sklearn.lda import LDA
#lda = LDA(n_components=2)
#dataobj.lda= lda.fit_transform(dataobj.features, np.ravel(dataobj.label_colarray))
#plt.scatter(dataobj.lda[:,0],dataobj.lda[:,1], c = dataobj.label_colarray)
#plt.show()


#TEST INDIVIDUAL CLASSIFIERS
print dataobj.features.shape
rf = RandomForest(no_trees = 50000)
scores = []
print 'training a random forest classifier!'
for i in range(10):
	classtester = ClassifierTester(dataobj.features,np.ravel(dataobj.label_colarray), training_test_split = 80)
	(score, predictedlabelsprobs, reallabels) = classtester.test_classifier(rf)
	print score, 'percent correct!'
	scores.append(score)
print '****',np.mean(scores), 'is mean score','****'


print 'training a support vector classifier'
for i in range(10):
    classtester = ClassifierTester(dataobj.features,np.ravel(dataobj.label_colarray), training_test_split = 80)
    svcclf = SupportVecClf(k_type = 'rbf')
    (score, predictedlabelsprobs, reallabels) = classtester.test_classifier(svcclf)
    print score, 'percent correct!'
    scores.append(score)
print '****', np.mean(scores), 'is mean score ****'

print 'training a K nearest neighbors classifier'
knn_clf = KNeighbors(15)
(score, predictedlabelsprobs, reallabels) = classtester.test_classifier(knn_clf)
print score, 'percent correct!'

plots.radviz(dataobj)
#plt.show()
#plots.scatter_matrix(dataobj)
#raw_input()
#plt.close('all')
#for row_index in range(predictedlabelsprobs.shape[0]):
 #   print predictedlabelsprobs[row_index,:], ' actual label was :', reallabels[row_index]