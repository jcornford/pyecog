from sklearn.svm import SVC
class SupportVecClf():
    """
    Support vector classifier
    """
    def __init__(self, k_type= 'rbf'):
        self.classifier = SVC()
    def fit(self, X, y):
        """
        Method to fit the model.
        
        Parameters:
        X : 2d numpy array of training data, X.shape = [n_samples, d_features]
        y : 1d numpy array of training labels
        """
        self.classifier = self.classifier.fit(X, y)
        

    def predict(self, X):
        """
        Method to apply the model data
        
        Parameters:
        X : 2d numpy array of test data
        """
        return self.classifier.predict(X)
    
    def test(self, X, y):
        """
        Method to apply the model data
        
        Parameters:
        X : 2d numpy array of test data
        y : actual labels
        """
        # [:, 1] to get the second column, which contains the probabilies of 
        # of class being 1
        score  =self.classifier.score(X,y)
        return score