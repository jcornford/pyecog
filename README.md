# networkclassifer
Program for classifying network state in a given window before light pulses.

Currently taking 10 seconds before a light pulse and extracting 22 features from this window.

Todo:

1. Produce pdfs of the events within the training an test/ anything that comes
2. Speed up feature extraction
3. Implement feature selection
4. Sort out the code from early days vs ipython notebook copies etc
5. Asses scaling, preprocessing (baseline subtraction etc) and imputing options.

#To think about:
What to do with the first 100ms of rolling window?

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
