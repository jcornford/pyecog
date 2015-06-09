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
