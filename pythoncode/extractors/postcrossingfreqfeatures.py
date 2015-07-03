import numpy as np
import featureBaseClass
import scipy.stats as st
import sklearn

class PostCrossingPower():
    """
    Extracting frequency features

    """

    def __init__(self):
        pass

    def extract(self, data):
        data/= np.std(data,axis = 1)[:,None] 
        #dataobj.data_array /= np.std(dataobj.data_array)
        mask = np.where(data[:,:-513]<-4,1,0)
        mask1 = np.roll(mask[:],1)
        mask = abs(np.subtract(mask,mask1))
    
        crossings = []
        for i in range(mask.shape[0]):
            crossings.append(np.where(mask[i,:])[0][1::2])

        myWaveletFamily, frequencies = self._custommorletFamily([5,10,35])

        fs = 512 
        window = int(fs*0.5)
        delay = 5
        allMeanPower = []

        for ti in range(data.shape[0]):
            segments = []
            secMeanPower = []
            traceMeanPower = []
            if len(crossings[ti])>0:
                for i in crossings[ti]:
                    segments.append(data[ti,i+delay:i+window])
                    convResults = self._convolve(data[ti,i+delay:i+window], myWaveletFamily,frequencies, fs = 512)
                    powerArray  = np.absolute(convResults)
                    secMeanPower.append(np.mean(powerArray,axis = 1)) # for features
                secMeanPower = np.array(secMeanPower)
                
                traceMeanPower.append(np.mean(secMeanPower,axis = 0))
            else:
                traceMeanPower.append(np.ones(len(frequencies)))
                    
            traceMeanPower = np.array(traceMeanPower)
            allMeanPower.append(traceMeanPower)
            
            

        allMeanPower = np.vstack(allMeanPower)
        self.names = frequencies
        return allMeanPower

        #features = np.log(np.absolute(np.fft.rfft(data, axis = 1)[:,1:200]))
        #features  = sklearn.preprocessing.scale(features, axis = 0)
        #print 'here'
        #return features


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

