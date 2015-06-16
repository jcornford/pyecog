# networkclassifer
Repositry to hold files for classifying network state in a given window before light pulses

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