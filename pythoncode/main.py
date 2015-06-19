import numpy as np

# Basic script to run network state classfication

from loadSeizureData  import LoadSeizureData
from classifierTester import ClassifierTester
from extractors.basicFeatures    import BasicFeatures
from classifiers.randomForestClassifier import RandomForest
from extractors.freqfeatures     import FreqFeatures
from relabeling_functions import relabel, reorder
from visualisation import plots

dirpath = '/Users/Jonathan/Documents/PhD /Seizure_related/Network_states/VMData/Classified'
dataobj = LoadSeizureData(dirpath)
dataobj.load_data()
dataobj = relabel(dataobj)
dataobj = reorder(dataobj)
basicStatsExtractor = BasicFeatures()
dataobj.extract_feature_array([basicStatsExtractor])

rf = RandomForest(no_trees = 100)
classtester = ClassifierTester(dataobj.features,np.ravel(dataobj.label_colarray), training_test_split = 80)
(score, predictedlabelsprobs, reallabels) = classtester.test_classifier(rf)
print 'training a random forest classifier!'
print score, 'percent correct!'

plots.radviz(dataobj)
plots.scatter_matrix(dataobj)
raw_input()
#for row_index in range(predictedlabelsprobs.shape[0]):
 #   print predictedlabelsprobs[row_index,:], ' actual label was :', reallabels[row_index]