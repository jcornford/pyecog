import numpy as np
class ClassfierTester():
    def __init__(self,features,labels):
        """
        features: 2d array
        labels: 1d array
        classifers: instance of an PredictorBaseClass subclass
        """
        self.features = features
        self.labels= labels
        
        
        #generate training and test sets
        self._gen_train_and_test()
        
    def _gen_train_and_test(self,percent = 80):
        
        self.rng_state = np.random.get_state()
        np.random.shuffle(self.features)
        np.random.set_state(self.rng_state)
        np.random.shuffle(self.labels)
        
        self.slicepoint = int((self.features.shape[0]/100.0)*percent)
        
        self.test_features = self.features[self.slicepoint:,:]
        self.test_labels = self.labels[self.slicepoint:]
        
        self.train_features = self.features[:self.slicepoint,:]
        self.train_labels = self.labels[:self.slicepoint]
        #for i in range(61):
            #print i,self.test_features[i,:3],self.test_labels[i]
        
        
    def test_classifier(self, classifier):
        self.classifier= classifier
        self.classifier.fit(self.train_features,self.train_labels)
        self.score = self.classifier.test(self.test_features,self.test_labels)
        self.predmatrix = self.classifier.predict(self.test_features)
        return self.score*100, self.predmatrix, self.test_labels