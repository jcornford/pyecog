# networkclassifer
Repositry to hold files for classifying network state in a given window before light pulses.

Currently taking 10 seconds before a light pulse ans extract features from this window.

1. Add something for after the feature matrix has been computed - to perform dimensionality reduction, LDA, z-norm, standardisation, play with rLDA etc. Remember that this will be fitted on the trainingset and then test set is transoformed.

2. Cross validation. 

3. Hyperparameter fitting. 

4. Std etc, be caureful on th etype of normalisation used before them.

5. Ways to see classification of individual states better

6. Classifier based on probabiity. 

## To do:
1. Write code to visualise and go through and check/change labels of the 10 second sections.
2. Analysis of states?

* As features are currently each 1d, write some visualisation code with features and their correct labels. 
	- a) scatterplot matrix  - pandas, how to customise?
	- b) projection onto two or three principal components?
	- c) Check out pandas radviz plotting algorithm?

* Work out validation method that we want
* How to assess the importance of features
* Normalise features to unit 1
* Some bugs in the orginal ephys trace - remove the glitches!
* Work out type of normalisation wanted, currently z norm. 
* Have an easy way to check the feature extraction on a small data set so can see the single value given at the end makes sense... 

## Possible features
* Stationarity testing
Potential features:
vector strength of cross freq coulping 
Wavelet coefficients
stationarity of prev to light - diff to when dive into blocks?
coastline
PCs of AR coefs, over time many small stationary bins
PCs in general - will not work?!
The low freq components? - including the weird change ups m hines did, per sec?
Or sum of wavelets
Eigenvalues

Asses by having counter for x and just plotting? per one. For AR "fingerprint" can use subplots.


Preprocessing?
::20,
0 mean and std varience? z  norm?
in blocks of whole thing?

Regularisation vs pca

mda, lda?

## 2015 06 21
1.Cross validation
2.split test dataset away and use at the very very end
when standerdising etc, do not include the test dataset
3.power after a std crossing? as a feature for class 2?
4.detect baseline sections in the whole trace? 0 crossings, stationarity of mean?