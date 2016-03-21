# networkclassifer

Scripts orginally for classifying network state in a given window before light pulses. Now undergoing major overhaul to
be more general, at to be able to convert "ndf" files. 

# Usage:
Scripts for detecting state before light pulse are found in "/pythoncode".  

network_classifier.py
 - for training the classifier, supply training data and labels
 - can output blank pdfs for labelling

network_predictor.py 
  - supply with unlabeled data, options for plotting predictions as pdf.
  - outputs excel file
 

## Todo:
1. Refactor network_loader - bug fix on the window and downsampling

## Possible features to implement
* Stationarity testing
* Potential features:
* Vector strength of cross freq coulping 
* AR coefs, paper putting them through a svm?
* Eigenvalues, hines used no?

