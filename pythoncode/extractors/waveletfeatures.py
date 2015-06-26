import numpy as np
import featureBaseClass
import scipy.stats as st
import sklearn

class WaveletFeatures():
    """
    Extracting frequency features using wavelets

    """

    def __init__(self):
        pass

    def extract(self,data):
        power = []
        self.myWaveletFamily, self.frequencies = self._custommorletFamily([2,6,10,20,40,100,150])
        for i in range(data.shape[0]):
            self.convResults = self._convolve(data[i,:], self.myWaveletFamily,self.frequencies, fs = 512)
            powerArray  = np.absolute(self.convResults)
            power.append(powerArray)
        self.max_wave_powers = np.vstack([np.mean(power[i], axis = 1) for i in range(len(power))])
        self.names = self.frequencies
        return self.max_wave_powers


    def _convolve(self,dataSection, wavelets,frequencies, fs):
        try:
            dataSection=dataSection[:,0]
        except:
            pass

        convResults = []

        for i in range(len(wavelets)):
            wavelet = wavelets[i]
            complexResult = np.convolve(dataSection,wavelet,'same')
            convResults.append(complexResult) 

        convResults = np.array(convResults) # convert to a numpy array!
        return convResults

    def _complexMorlet(self,freq, samplingFreq, timewindow = (-0.1,0.1), noWaveletCycles=6,
                    freqNorm = False,
                    plotWavelet = False):
        '''
        See above markdown cell for formula ;)
        
        freq              = wavelet frequency
        samplingFrequency = 'insert data sampling frequency here'
        noWaveletCycles   = 'reread variable name' - trades off temp and freq precision
        timewindow        = timewindow over which to calculate wavelet
        freqNorm          = Bool, decide whether to normlise over frequencies
        plotWavelet       = Plot the construction?
        '''

        n = noWaveletCycles
        f = freq
        timeaxis = np.linspace(timewindow[0],timewindow[1],samplingFreq*(2.*timewindow[1]))
        sine_wave = np.exp(2*np.pi*1j*f*timeaxis)
        gstd = n/(2.*np.pi*f)
        gauss_win = np.exp(-timeaxis**2./(2.*gstd**2.))
        
        A = 1 # just 1 if not normlising freqs
        if freqNorm:
            #gstd    = (4/(2*np.pi*f))**2# this is his?
            A = 1.0/np.sqrt((gstd*np.sqrt(np.pi)))
            print 'freqNorm is :',A,' for', f, ' Hz'
        
        wavelet = A*sine_wave*gauss_win
        #print wavelet.shape
            
        return wavelet, timeaxis

            
    def _custommorletFamily(self,central_freqs,timewindow = (-0.1,0.1),step = 5,fs = 512.0, freqNorm = False):
        '''
        Returns a list of wavelets 
        Inputs:
        start, stop and step of the frequencies
        fs is sampling freq - should equal data's
        freqNorm - to normlise over frequencies?
        '''
        waveletList = []
        frequencies = []
        for f in central_freqs:
            wavelet, xaxis = self._complexMorlet(f,fs,timewindow ,freqNorm=freqNorm)
            waveletList.append(wavelet)
            frequencies.append(f)
        return waveletList,frequencies