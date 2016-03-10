# networkclassifer

Scripts orginally for classifying network state in a given window before light pulses.

# Usage:
Scripts for detecting state before light pulse. These are found in /pythoncode
network_classifier - for training the classifier
network_predictor 
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

