import numpy as np
import scipy.misc
import scipy.signal
import neurokit2 as nk
import os
from sklearn import preprocessing


import pandas as pd
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
# import timesynth as ts
from neurokit2.misc import NeuroKitWarning, listify
from itertools import tee, chain, cycle, groupby
from scipy import signal
from statistics import stdev, mean


def body_len(mal_arr, interval):
    body_len_arr = []
    for i in range(0, len(mal_arr), interval):
        body_len_arr.append(np.nanmean(mal_arr[i:i + interval]))
    return body_len_arr


def find_decreaingV2(ll, num_point=10):
    found = [[] for i in range(len(ll) * 10)]  # create an  empty list of lists (with extra free spaces just in  case)
    peak_ind = 0
    for j in range(len(found)):
        for i in range(j, len(ll) - 1):
            if ll[i] > ll[i + 1]:
                found[peak_ind].append(ll[i])
            else:
                if i + 1 < len(ll) - 1:
                    found[peak_ind].append(ll[i])
                peak_ind += 1
        found_cleaned = [x for x in found if len(x) >= num_point]  # remove the sets w fewer than 6# peaks
        if len(found_cleaned) >= 3:
            found.sort(key=len, reverse=True)
            return found_cleaned


def find_increaingV2(ll, num_point=10):
    found = [[] for i in range(len(ll) * 10)]
    peak_ind = 0
    for j in range(len(found)):   # for every potential starting seq number
        for i in range(0, len(ll) - 1):
            if ll[i] <= ll[i + 1]:
                found[peak_ind].append(ll[i])
            else:
                if i < len(ll) - 1:
                    found[peak_ind].append(ll[i])
                peak_ind += 1
        found_cleaned = [x for x in found if len(x) >= num_point]  # remove the sets w fewer than 6# peaks
        if len(found_cleaned) >= 3:
            found.sort(key=len, reverse=True)  # return a list of lists sorted my lens of its elems in descending order
            return found_cleaned


"""" 
stop_time = 10
num_points = stop_time * 50
period = 2
frequency = 1/period
# Initializing TimeSampler
time_sampler = ts.TimeSampler(stop_time=stop_time)
# Sampling irregular time samples
irregular_time_samples = time_sampler.sample_irregular_time(num_points=num_points, keep_percentage=100)

sinusoid = ts.signals.PseudoPeriodic(amplitude = 1, frequency=1, ampSD = 0.1, freqSD = 0.1)
timeseries = ts.TimeSeries(sinusoid)

samples, signals, errors = timeseries.sample(irregular_time_samples)
# Plotting the series
#plt.plot(irregular_time_samples, samples, marker='o', markersize=4)
#plt.xlabel('Time')
#plt.ylabel('Magnitude')
#plt.ylim([-amplitude-10, amplitude+10])
#plt.title('Irregularly sampled sinusoid with noise');
"""

"""
check if a lag is far/different enough from other lags in list_of_lags
num = num of reference
leeway = +/- frames exclusion 
list_of_lags: list of lag timestamps corresponding to timestamps of the left vallyes of
  the first peak in a 3-peak sequence  
"""


def check_far_enough(lag, leeway, list_of_lags):
    result = True
    for ele in list_of_lags:
        if lag < ele + leeway and lag > ele - leeway:  # element
            # print("element", ele, "is too close to num", lag)
            result = False
            break
        # else:
        # print("element", ele, "is far enough from to num", lag)
    return result


## todo: deal with negative indexes?
# goal_num == how many peak sets do we want to identify
# leeway == what's the min distance between different peak sets
def remove_overlapping(corr, goal_num_sets, signal_unpadded, leeway=5):
    k = goal_num_sets * 2
    sorted_corrs = np.argsort(corr)  # indexes of highest corrs in order smallest -> highest
    curr_top_corr_inds = list(reversed(list(sorted_corrs[-k:])))  # get top k best correlated inds
    # print("the highest corr ind", list(max_corr_inds))
    max_corr_inds = []
    counter = 0  # counter of how many good distinct peak sets were identified
    while counter < goal_num_sets:
        for i in curr_top_corr_inds:  # get a lag for each highest correlation
            lag = i  #not actually a lag; lag is i+len(signal_unpadded)
            if lag>300 and check_far_enough(lag, leeway=leeway,
                                list_of_lags=max_corr_inds):  # if this lag is not to close to any other lags
                if lag > len(signal_unpadded) and lag < len(corr) - 2 * len(signal_unpadded):
                    max_corr_inds.append(lag)
                    counter += 1
        # sorted_corrs = np.argsort(sorted_corrs)
        sorted_corrs = sorted_corrs[:-k]  # remove the idexes we already looked through
        curr_top_corr_inds = list(reversed(list(sorted_corrs[-k:])))  # get top k best correlated inds
        #print("the highest corr ind", list(curr_top_corr_inds))
    return max_corr_inds


"""
duration : float, Desired length of duration (s).
sampling_rate : int, the desired sampling rate (in Hz, i.e., samples/second).
frequency : float or list, oscillatory frequency of the signal (in Hz, i.e., oscillations per second). == 1/period 
amplitude : float or list, Amplitude of the oscillations.
   """
# av_worm_len = mean(currMAL)
def cross_correlate(currMAL, fps=5,  freq_elong = 0.6, freq_contr = 0.8, goal_num_sets=10, leeway=5):

    scaling_factor = 0.8
    av_worm_len = np.nanmean(currMAL) * scaling_factor #todo: change?
    amplitude = (av_worm_len / 2)* scaling_factor
    sampling_rate = fps * 1.6 # arbitrary
    #freq_elong = 0.6  # --> duration of elong ~ (1/0.6)/2 = 0.833 sec
    #freq_contr = 0.8  # scrunch/sec  --> duration of contraction ~ (1/0.8)/2 = 0.625 sec
    duration = (freq_elong + freq_contr) * 5
    # The temporal frequency was converted into a relative speed in body lengths per second by defining
    # v_m* = v_m ⋅ |delta e_max|  (where e is the max in-cycle length change, normalize by max gliding length)
    # for D. japonica, oscillation frequency is between 0.7 and 0.75 Hz
    elong = nk.signal_simulate(duration=duration, sampling_rate=sampling_rate, frequency=[freq_elong],
                               amplitude=[amplitude])
    contr = nk.signal_simulate(duration=duration, sampling_rate=sampling_rate, frequency=[freq_contr],
                               amplitude=[amplitude])

    elong_intervals = find_increaingV2(elong, num_point=int((1 / freq_elong) / 2 * fps))
    contr_intervals = find_decreaingV2(contr, num_point=int((1 / freq_contr) / 2 * fps))

    signal_unpadded = []
    for i in range(1):
        midway = int(len(elong_intervals[0])/2)
        signal_unpadded.append(elong_intervals[0][midway:])
        signal_unpadded.append(contr_intervals[0])
        signal_unpadded.append(elong_intervals[1])
        signal_unpadded.append(contr_intervals[1])
        signal_unpadded.append(elong_intervals[2])
        midway = int(len(contr_intervals[0])/2)
        signal_unpadded.append(contr_intervals[2][:midway+1])


    signal_unpadded = list(chain.from_iterable(
        signal_unpadded)) # unpack the list of list into one long list containing all y coords of the data
    #plt.plot(signal_unpadded)


    # create a padded version of the signal
    signal_padded = np.empty(len(currMAL))

    signal_padded.fill(av_worm_len)  # shift the axis up so that the signal oscillates around y=av worm length
    signal_padded[:len(signal_unpadded)] = signal_unpadded + av_worm_len
    #signal_unpadded = [x+av_worm_len*0.7 for x in signal_unpadded]

    #currMAL = list(chain.from_iterable(preprocessing.normalize([currMAL])))  #this was an attempt to
    #signal_unpadded = list(chain.from_iterable(preprocessing.normalize([signal_unpadded])))
    #signal_padded = list(chain.from_iterable(preprocessing.normalize([signal_padded])))

    corr = signal.correlate(currMAL, signal_unpadded, mode="full")
    #corr = signal.correlate(currMAL, signal_padded, mode="same")

    #print("mean of top 5 corrs", mean(sorted(corr, reverse=True)[:5]))
    lags = signal.correlation_lags(len(currMAL), len(signal_unpadded), mode="full")  # == corr[i] - len(signal_unpadded)
    max_corrs = remove_overlapping(corr, goal_num_sets=goal_num_sets, signal_unpadded=signal_unpadded, leeway=leeway)
    # timestamps of left valleys of the first peak for sets of 3 peaks
    list_of_lags_final = lags[max_corrs]


    return list_of_lags_final, signal_unpadded


def generate_overlap_plots(list_of_lags_final, currMAL, signal_unpadded, filepath, wellNum):
    av_worm_len = np.nanmean(currMAL)
    for i in list_of_lags_final:
        # print("lag ", i)
        overlap = np.empty(len(currMAL))
        overlap.fill(av_worm_len)  # shift the axis up so that the signal oscillates around y=av worm length
        overlap[i:i + len(signal_unpadded)] = signal_unpadded + av_worm_len
        plt.plot(overlap)  # visualize the overlap
        plt.plot(currMAL)
        plt.title("Well "+ str(wellNum))
        plt.xlim([i - 100, i + len(signal_unpadded) + 100])
        outpath = os.path.expanduser(filepath + "/results/synth_signal_tests/peak overlaps well" + str(wellNum) + "ind" + str(i) + ".png")
        plt.savefig(outpath)
        plt.show()
        plt.close()


""" 
wellNum = 5

filepath = "/Users/Arina/Desktop"
# filepath = "/Volumes/Collins_Lab/15"
filename = filepath + "/results/well_data/MAL_well" + str(wellNum) + ".csv"
currMAL = genfromtxt(filename, delimiter=',')

list_of_lags_final, signal_unpadded = cross_correlate(currMAL, fps=10)
generate_overlap_plots(list_of_lags_final, currMAL, signal_unpadded, filepath, wellNum)
"""