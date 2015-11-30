import time
import pickle
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor():

    def __init__(self, dataset):
        self.dataset = dataset

        window_size = 100
        array_std = np.std(self.rolling_window(self.dataset,window=window_size),-1)
        array_window = np.zeros([self.dataset.shape[0],window_size-1])
        rolling_std_array = np.hstack((array_window,array_std))
        masked_std = np.ma.masked_where(rolling_std_array > threshold, rolling_std_array)

    @ staticmethod
    def rolling_window(array, window):
        """
        Remember that the rolling window actually starts at the window length in.
        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)



def rolling_std(X, w):

    print time.clock(),'rolling std start'
    r = np.empty(X.shape)
    r.fill(np.nan)
    for i in range(w - 1, X.shape[0]):
        r[i] = np.std(X[(i-w+1):i+1])
    print time.clock(),'rolling std end'
    return r

def rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def extract_features(dataset):


    baseline_point_subtraction = True
    start  = time.clock()
    threshold = 0.04
    window_size = 100
    array_std = np.std(rolling_window(dataset,window=window_size),-1)
    array_window = np.zeros([dataset.shape[0],window_size-1])
    rolling_std_array = np.hstack((array_window,array_std))
    masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, dataset)

    mean_baseline_vector = np.mean(masked_std_below_threshold,axis = 1)
    if baseline_point_subtraction:
        dataset_after_subtraction_option = dataset - mean_baseline_vector[:,None]
    masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, dataset_after_subtraction_option)
    event_dataset = [np.ma.compressed(masked_std_above_threshold[i,:]) for i in xrange(dataset.shape[0])]

    baseline_length = [len(np.ma.compressed(masked_std_below_threshold[i,:])) for i in xrange(dataset.shape[0])]

    indexes = np.array(np.arange(5120))
    #baseline_diff = [indexes[np.logical_not(masked_std_below_threshold[i].mask)] for i in xrange(dataset.shape[0])]
    baseline_mean_diff = [np.mean(np.diff(indexes[np.logical_not(masked_std_below_threshold[i].mask)])) for i in xrange(dataset.shape[0])]


    #Now for the stats on the event dataset
    print("Extracting stats on event dataset..."),
    import scipy.stats as st
    min_ = []
    max_ = []
    mean_ = []
    std_ = []
    skew_ = []
    kurtosis_ = []
    diff_sum_ = []

    for trace in event_dataset:
        if trace.shape[0]:
            min_.append(np.min(trace))
            max_.append(np.max(trace))
            mean_.append(np.mean(trace))
            std_.append(np.std(trace))
            kurtosis_.append(st.kurtosis(trace))
            skew_.append(st.skew(trace))
            diff_sum_.append(np.sum(np.absolute(np.diff(trace))))
        else:
            min_.append(np.NaN)
            max_.append(np.NaN)
            mean_.append(np.NaN)
            std_.append(np.NaN)
            kurtosis_.append(np.NaN)
            skew_.append(np.NaN)
            diff_sum_.append(np.NaN)

    min_ = np.array(min_)
    max_ = np.array(max_)
    mean_ = np.array(mean_)
    std_ = np.array(std_)
    skew_ = np.array(skew_)
    kurtosis_ = np.array(kurtosis_)
    diff_sum_ = np.array(diff_sum_)
    print("DONE")

    print ("Extracting peak/valley features..."),
    from numpy import NaN, Inf, arange, isscalar, asarray, array
    def peakdet(v, delta, x = None):
        """
        function [maxtab, mintab]=peakdet(v, delta, x)
                [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                maxima and minima ("peaks") in the vector V.
                MAXTAB and MINTAB consists of two columns. Column 1
                contains indices in V, and column 2 the found values.

                With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
                in MAXTAB and MINTAB are replaced with the corresponding
               X-values.

                A point is considered a maximum peak if it has the maximal
                value, and was preceded (to the left) by a value lower by
                DELTA.
        """
        maxtab = []
        mintab = []
        if x is None:
            x = arange(len(v))
        v = asarray(v)
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        lookformax = True
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        return array(maxtab), array(mintab)

    pk_nlist = []
    val_nlist = []
    for i in range(dataset.shape[0]):
        pk, val = peakdet(dataset[i,:], 0.5)
        pk_nlist.append(pk)
        val_nlist.append(val)

    # always 2 columns
    n_pks = []
    n_vals = []
    av_pk = []
    av_val = []
    av_range = []
    for i in range(len(pk_nlist)):
        n_pks.append(pk_nlist[i].shape[0])
        n_vals.append(val_nlist[i].shape[0])

        if pk_nlist[i].shape[0]:
            av_pk.append(np.mean(pk_nlist[i][:,1]))
        else:
            av_pk.append(np.NaN)

        if val_nlist[i].shape[0]:
            av_val.append(np.mean(val_nlist[i][:,1]))
        else:
            av_val.append(np.NaN)

        if val_nlist[i].shape[0] and pk_nlist[i].shape[0]:
            av_range.append(
                abs(np.mean(pk_nlist[i][:,1])+np.mean(val_nlist[i][:,1]))
                )
        else:
            av_range.append(np.NaN)



    n_pks = np.array(n_pks)
    n_vals = np.array(n_vals)
    av_pk = np.array(av_pk)
    av_val = np.array(av_val)
    av_range = np.array(av_range)
    print("DONE")

    print("Extracting wavelet features from event dataset..."),
    def complexMorlet(freq, samplingFreq, timewindow,
                  noWaveletCycles=6,freqNorm = False):
        '''
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

    def convolve(dataSection, wavelets):
        convResults = []
        for i in range(len(wavelets)):
            wavelet = wavelets[i]
            complexResult = np.convolve(dataSection,wavelet,'same')
            convResults.append(complexResult)

        convResults = np.array(convResults) # convert to a numpy array!
        #print convResults.shape
        return convResults

    fs = 512
    window = int(fs*0.5)


    waveletList = []
    frequencies = [1,5,10,15,20,30,60, 90]
    for freq in frequencies:
        wavelet, xaxis = complexMorlet(freq,fs,timewindow = (-0.1,0.1))
        waveletList.append(wavelet)

    power = []
    meanpower = []
    for i in range(len(event_dataset)):
        if event_dataset[i].shape[0]:
            convResults = convolve(event_dataset[i], waveletList)
            powerArray  = np.absolute(convResults)
            #print powerArray.shape
            power.append(powerArray)
            meanpower.append(np.mean(powerArray,axis = 1))

        else:
            power.append(np.ones((len(frequencies)))*np.NaN)
            meanpower.append(np.ones((len(frequencies)))*np.NaN)
    meanpower = np.array(meanpower)
    print("DONE")

    print("Really need post crossing power")

    print("Stacking up...")
    event_stats = np.vstack([min_,max_,mean_,skew_,std_,kurtosis_,diff_sum_]).T
    print event_stats.shape
    baseline_features = np.vstack([baseline_length,baseline_mean_diff]).T
    print baseline_features.shape
    pkval_stats = np.vstack([n_pks,n_vals,av_pk,av_val,av_range]).T
    print pkval_stats.shape

    print meanpower.shape

    feature_array = np.hstack([event_stats,baseline_features,
                               pkval_stats,meanpower])
    print feature_array.shape
    print 'DONE!',
    print 'took ', time.clock()-start, ' seconds'
    return feature_array
start = time.clock()
dataset = pickle.load(open('../dataset','rb'))
print time.clock()-start, 'seconds to load - now running extraction'
start2 = time.clock()
extract_features(dataset)
print time.clock()-start2, 'finished extraction'

# begin piss-around
'''
def rolling_std(X, w):
    print time.clock(),'rolling std start'
    r = np.empty(X.shape)
    r.fill(np.nan)
    for i in range(w - 1, X.shape[0]):
        r[i] = np.std(X[(i-w+1):i+1])
    print time.clock(),'rolling std end'
    return r

def rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

r_std = rolling_std(dataset[6,:], 100)
threshold = 0.04
bis = np.where(r_std<threshold)[0]
#plt.plot(r_std, 'r')


print array_std.shape
print array_window.shape
rolling_std_array = np.hstack((array_window,array_std))
print rolling_std_array.shape

plt.plot(rolling_std_array[6,:], 'k')

plt.plot(dataset[6,:],'g')
plt.plot(np.arange(5120)[r_std<threshold],dataset[6,:][r_std<threshold],'rx')
plt.plot(np.arange(5120)[rolling_std_array[6,:]<threshold],dataset[6,:][rolling_std_array[6,:]<threshold],'kx')

print r_std.shape

baseline_point_subtraction = True
start  = time.clock()
threshold = 0.04
window_size = 100
array_std = np.std(rolling_window(dataset,window=window_size),-1)
array_window = np.zeros([dataset.shape[0],window_size-1])
rolling_std_array = np.hstack((array_window,array_std))
masked_std_below_threshold = np.ma.masked_where(rolling_std_array > threshold, dataset)

mean_baseline_vector = np.mean(masked_std_below_threshold,axis = 1)
if baseline_point_subtraction:
    dataset_after_subtraction_option = dataset - mean_baseline_vector[:,None]
masked_std_above_threshold = np.ma.masked_where(rolling_std_array < threshold, dataset_after_subtraction_option)
event_dataset = [np.ma.compressed(masked_std_above_threshold[i,:]) for i in xrange(dataset.shape[0])]

baseline_length = [len(np.ma.compressed(masked_std_below_threshold[i,:])) for i in xrange(dataset.shape[0])]

indexes = np.array(np.arange(5120))
baseline_diff = [indexes[np.logical_not(masked_std_below_threshold[i].mask)] for i in xrange(dataset.shape[0])]



#print baseline_diff_pre.shape, 'is diff_shape'
# need to grab baseine length, and also baseline

print baseline_length[6], 'is baseline len of 6'
for i in xrange(100):
    print baseline_diff[6][i+50]
    #print baseline_diff[6,i], 'is baseline diff'
    print masked_std_below_threshold[6,i+50],
    #print 'diff is', baseline_diff_pre[6,i+50]

baseline_dp = []
    for i in range(dataset.shape[0]):
        rstdi = rolling_std(dataset[i,:], 100,)# thresh = 0.04)
        e_index = np.where(rstdi>threshold)[0]
        b_index = np.where(rstdi<threshold)[0]
        baseline_dp.append(b_index)


    baseline_length = []
    baseline_mean_diff = []

    for dp_is in baseline_dp:
        blen_sec = dp_is.shape[0]/512.
        baseline_length.append(blen_sec)

        blen_diff = np.diff(dp_is)
        baseline_mean_diff.append(np.mean(blen_diff))

    baseline_length = np.array(baseline_length)
    baseline_mean_diff = np.array(baseline_mean_diff)
    print("Done baseline len")
    print time.clock()

print 'to do rolling etc, took ', time.clock() - start, 'seconds'
print rolling_std_array.shape

print mean_baseline_vector[6]

print mean_baseline_vector.shape
plt.plot(dataset_after_subtraction_option[6,:],'b')
#plt.plot(masked_std_below_threshold[6,:],'r')
plt.plot(masked_std_above_threshold[6,:],'kx')
plt.plot(event_dataset[6],'g')
#plt.plot(baseline_diff[6],'r')
plt.show()
'''

