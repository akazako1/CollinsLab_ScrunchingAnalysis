import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # Apple doesn't like Tkinter (TkAgg backend) so I needed to change the backend to 'Qt5Agg'
import statsmodels.api as sm
import numpy as np
import os
import pandas as pd
from numpy import genfromtxt
from os import makedirs
from os.path import exists
from itertools import combinations, chain
from scipy.signal import find_peaks, peak_prominences, peak_widths
from statistics import stdev, mean
import generate_synth_signal as synth_signal


def read_MAL_data(wellNum=None, filename=None):
    if wellNum:
        filename = "/Users/Arina/PycharmProjects/ScrunchingTrack/MAL data well" + str(wellNum) + ".csv"
        # /Users/Arina/PycharmProjects/ScrunchingTrack/MAL data well1.csv
    my_data = genfromtxt(filename, delimiter=',')
    return my_data


def ind_exists(sset, ind):
    for i in range(1, len(sset)):
        if sset[i] == ind:
            return True
    else:
        return False


"""  Get all combinations of (sequential) peaks in set of peaks with >3 peaks"""
def get_combinations(sset, mode="sequential"):  # todo: check that other conditions still hold true when we get combinations
    ssets = []
    if len(sset) > 3:
        if mode=="sequential":
            for i in range(len(sset)-2):
                ssets.append(sset[i:i+3])
        elif mode=="any":
            ssets = list(combinations(sset, 3))
    else:
        ssets = [sset]
    return ssets

# takes in a list of ssets (times) and checks that they are not too far apart
def check_not_too_far(ssets, peak_data):
    ssets_cleaned = []
    for sset in ssets:
        sset_times = to_timestamps(sset, peak_data)
        if sset_times[2]-sset_times[1]<25 and sset_times[1]-sset_times[0]<25:
            ssets_cleaned.append(sset)
    return list(ssets_cleaned)

""" Converts a set w peak indexes from the data table to peak timestamps (indexes in the MAL array) """
def to_timestamps(sset, peak_data) -> object:
    peak_set_times = []
    for ind in sset:
        ind = int(ind)
        peak_set_times.append(int(peak_data[ind][1]))
    return list(peak_set_times)


def Lowess(data, pts=6, itn=3, order=1):
    data = pd.DataFrame(data)
    x = np.array(data.index, dtype=float)
    # condition x-values to be between 0 and 1 to reduce errors in linalg
    x = x - x.min()
    x = x / x.max()
    y = data.values
    n = len(data)
    r = int(np.min([pts, n]))
    r = min([r, n - 1])
    order = max([1, order])
    # Create matrix of 1, x, x**2, x**3, etc, by row
    xm = np.array([x ** j for j in range(order + 1)])
    # Create weight matrix, one column per data point
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    # Set up output
    yEst = np.zeros(n)
    delta = np.ones(n)  # Additional weights for iterations
    for iteration in range(itn):
        for i in range(n):
            weights = delta * w[:, i]
            xw = np.array([weights * x ** j for j in range(order + 1)])
            b = xw.dot(y)
            a = xw.dot(xm.T)
            beta = np.linalg.solve(a, b)
            yEst[i] = sum([beta[j] * x[i] ** j for j in range(order + 1)])
        # Set up weights to reduce effect of outlier points on next iteration
        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return pd.Series(yEst, index=data.index, name='Trend')


def frac_lowess(mal_arr, frac=0.004, displayPlot=False):  # higher fraction ==more smoothing
    time = np.arange(start=0, stop=(len(mal_arr)) / 10, step=0.1)
    x = time
    y = mal_arr
    lowess = sm.nonparametric.lowess(y, x, frac=frac, missing='none', it=0)
    mal_smoothed = lowess[:, 1]
    """
    if displayPlot == True:
        plt.plot(x, y, '+')
        plt.plot(lowess[:, 0], lowess[:, 1])
        plt.show()
     """
    return mal_smoothed


def plot_valleys(mal_arr, xlabel=None):
    inds, dict = find_good_peaks(mal_arr)
    inds_right = dict["right_bases"]
    inds_left= dict["left_bases"]
    plt.plot(mal_arr)
    plt.plot(inds, np.array(mal_arr)[inds], "o")
    plt.plot(inds_right, np.array(mal_arr)[inds_right], "X")
    plt.plot(inds_left, np.array(mal_arr)[inds_left], "x")
    plt.legend(["", "peak", "right valley", "left valley"])
    if xlabel:
        plt.xlabel(xlabel)
    plt.close(0)


def plot_valleys_naive(mal_arr):
    peak_inds, dict = find_good_peaks(mal_arr)
    valley_inds, dict_neg = find_good_peaks(-mal_arr)
    plt.plot(mal_arr)
    plt.plot(peak_inds, np.array(mal_arr)[peak_inds], "o")
    plt.plot(valley_inds, np.array(mal_arr)[valley_inds], "x")
    plt.legend(["", "peaks", "valleys"])
    plt.close(0)
    return peak_inds, valley_inds


def match_peaks_valleys(peak, list_of_valleys):
    if peak<list_of_valleys[0]:     # check if there is no left valley:
        return 0, list_of_valleys[0]
    if peak>list_of_valleys[-1]:      # check if there is no right valley:
        return list_of_valleys[-1], peak+10
    for i in range(len(list_of_valleys)):
        if list_of_valleys[i]<peak and list_of_valleys[i+1]>peak:
            left_valley = list_of_valleys[i]
            right_valley = list_of_valleys[i+1]
            return left_valley, right_valley


def plot_valleys_prominences(sset, peak_data, currMAL):
    plt.plot(currMAL)
    #for ind in sset:
    #    plt.plot(peak_data[ind][1], peak_data[ind][10], "X")
    times = to_timestamps(sset, peak_data)
    _, inds_left, inds_right = peak_prominences(currMAL, times, wlen=10)
    print("left",inds_left )
    print("right", inds_right)
    plt.plot(times, np.array(currMAL)[times], "o")
    plt.plot(inds_right, np.array(currMAL)[inds_right], "X")
    plt.plot(inds_left, np.array(currMAL)[inds_left], "x")
    plt.xlim(times[0]-50, times[-1]+50)
    plt.legend(["", "peak", "right valley", "left valley"])
    plt.close(0)




""" Find all peaks (local MINIMA) without any filter """
def find_all_peaks(mal_arr, show=False):
    peakinds, _ = find_peaks(mal_arr)
    if show:
        plt.plot(mal_arr)
        plt.plot(peakinds, np.array(mal_arr)[peakinds], "x")
        plt.show()
        plt.close(0)
    return list(peakinds)

"""" 
def find_right_valleys(mal_arr, peak_data):
    peak_inds, peak_dict = find_good_peaks(mal_arr)
    valley_inds, _ = find_good_peaks(-mal_arr)
    for peak_ind in peak_inds:
        diff = valley_inds - peak_ind
        mask_right = np.ma.masked_less_equal(diff, 0) #mask the negative differences and zero since we are looking for values above
        #mask_left = np.ma.masked_greater_equal(diff, 0)
        if not np.all(mask_right):
            masked_diff_right = np.ma.masked_array(diff, mask_right)
            print("peak ind:", peak_ind, "valley ind:", masked_diff_right.argmin())
    # get left valley
"""



""" Perform peak (local MINIMA) filtering based on the following rules: 
 2) peak width larger than 1.75
 6) peak prominence smaller than 67 (or higher for larger worms)
 Returns  peakinds: indexes of all peaks that satisfy the condition; 
 peak dict: dictonary of features describing the peaks 
"""
def find_good_peaks(mal_arr, show=False, outpath=None, fps=5, pix_adj=1.5):
    fps_adj = fps / 5
    worm_size_adj = 1
    # todo: worm_size_adj =  XXXX/worm_size
    peakinds, peak_dict = find_peaks(mal_arr, distance=3, prominence=(-100, 67 * pix_adj * worm_size_adj), width=1.7*fps_adj)
    if show:
        plt.plot(mal_arr)
        plt.plot(peakinds, np.array(mal_arr)[peakinds], "x")
        if outpath is not None:
            plt.savefig(outpath)
            plt.show()
        plt.close(0)
    return peakinds, peak_dict


"""
Creates a peak_data data table containing information about the peaks found in find_good_peaks function
    1st column: reference index of the peak in this data table (Note: different than in the original script)
    2nd column: location (time) of the peak;
    3rd column: distance with the previous peak
    4th column: number of peaks withing one oscillation (added later)
    5th column: width of the peak; 
    6th column: prominence of the peak; 
    7th column: distance of the peak to the previous valley (lowest point between two peaks) == 'left_bases'
    8th column: distance of the peak to the next valley == 'right_bases'
    9th column: value of the previous valley
    10th column: difference of values of peak and its previous valley
    11th column: difference of values of peak and its next valley (Note: different than in the original script)
    12th: dict where key: set index, value:COM when the worms first started moving (Added later)
"""
def get_peak_data(mal_arr):
    peakinds, peak_dict = find_good_peaks(mal_arr)
    # get all peaks info
    peak_data = np.zeros([len(peakinds), 14], dtype=object)  # initialize the array to store peak data
    for i in range(len(peakinds)):
        # 1: length of the peak -- here index in this data table (for future reference)
        peak_data[i][0] = int(i)
        # 2: location (time) of the peak (in frames)
        peak_data[i][1] = int(peakinds[i])
        # 3rd column: distance to the previous peak
        if i > 0:
            peak_data[i][2] = peak_data[i][1] - peak_data[i - 1][1]
        # 5th column: width of the peak
        peak_data[i][4] = peak_dict['widths'][i]  # todo: unnecessary bc already filtered out wide enough peaks?

        # 6th column: prominence of the peak
        peak_data[i][5] = peak_dict['prominences'][i]
        peak_data[i][6] = peakinds[i] - peak_dict['left_bases'][i]  # 7th: distance to the previous valley (in frames)
        peak_data[i][7] = peak_dict['right_bases'][i] - peakinds[i]  # 8th: distance to the next valley (in frames)
        # 9th column: value of the previous valley
        peak_data[i][8] = mal_arr[peak_dict['left_bases'][i]]
        # 10th column: difference of values of peak and its previous valley
        peak_data[i][9] = mal_arr[peakinds[i]] - peak_data[i][8]
        # !! CHANGED !! 11th column: difference of values of peak and its next valley
        peak_data[i][10] = mal_arr[peakinds[i]] - mal_arr[peak_dict['right_bases'][i]]
        # 12th: dict where key: set index, value:COM when the worms first started moving
        peak_data[i][11] = {}
        # 13th: com
        peak_data[i][12] = []  #pl
        peak_data[i][13] = mal_arr[peak_dict['right_bases'][i]]

    return peakinds, peak_dict, peak_data


def add_valley_info():

    peak_data[i][5], inds_left, inds_right = peak_prominences(mal_arr, [peak_data[i][1]], wlen=10)
    # 6th column: prominence of the peak
    peak_data[i][5] = peak_dict['prominences'][i]
    peak_data[i][6] = peakinds[i] - peak_dict['left_bases'][i]  # 7th: distance to the previous valley (in frames)
    peak_data[i][7] = peak_dict['right_bases'][i] - peakinds[i]  # 8th: distance to the next valley (in frames)
    # 9th column: value of the previous valley
    peak_data[i][8] = mal_arr[peak_dict['left_bases'][i]]
    # 10th column: difference of values of peak and its previous valley
    peak_data[i][9] = mal_arr[peakinds[i]] - peak_data[i][8]
    # !! CHANGED !! 11th column: difference of values of peak and its next valley
    peak_data[i][10] = mal_arr[peakinds[i]] - mal_arr[peak_dict['right_bases'][i]]


""" Check that a given peak satisfies the following rules:
 3) 0.615<(peak value - previous valley value)/distance between the two values <6.5
 4) distance between peak and its previous valley >2
 5) distance of its two neighbor valleys >4
"""
def verify_good_peak(peak_data, peak_data_ind, fps=5, pix_adj=1.5):
    fps_adj = fps / 5
    if 0.615 * fps_adj / pix_adj < (
            peak_data[peak_data_ind][9] / peak_data[peak_data_ind][6]) < 6.5 * fps_adj / pix_adj:  # 3)
        if peak_data[peak_data_ind][6] > 2 * fps_adj:  # 4)
            if (peak_data[peak_data_ind][7] + peak_data[peak_data_ind][6]) > 4 * fps_adj:  # 5)
                return True
    return False


""" Collects info about all potential peak sets by looking through all peaks that fit the "good peak" rules """
def get_peak_sets(peak_data, fps=5, pix_adj=1.5):
    fps_adj = fps / 5
    good_peak_sets = [[] for x in range(peak_data.shape[0] + 1)]  # preallocate space
    for i in range(0, peak_data.shape[0] - 3):  # i here is the timestamp of the peak
        pks_set = good_peak_sets[i]
        if verify_good_peak(peak_data, i, fps=fps, pix_adj=pix_adj):  # verify that the first peak is good
            pks_set.append(peak_data[i][0])  # add the peak ind (in the data frame) to the list of potential sets
            next_ind = i + 1
            while next_ind < (len(peak_data) - 1) and (peak_data[next_ind][1] - peak_data[i][1]) < 23 * fps_adj:  # the distance between the peaks <= 23 frames
                if 4 * fps_adj <= (peak_data[next_ind][1] - peak_data[i][1]):  # the distance between the peaks  >=4 frames
                    if not ind_exists(pks_set, ind=peak_data[next_ind][0]) and verify_good_peak(peak_data, next_ind, fps=fps):
                        pks_set.append(peak_data[next_ind][0])  # add the index of the dataframe corr to the peak to the set
                next_ind += 1
    good_peak_sets = [x for x in good_peak_sets if len(x) >= 2]  # remove the sets w fewer than 2 peaks
    for pks_set in good_peak_sets:  # now go through the list of peak lists again and look for the third/next peak
        for j in range(1, len(pks_set)):  # for every elem of the set
            curr_ind = pks_set[j]  # index of the peak in the set that we are currently considering
            next_ind = curr_ind + 1
            while next_ind < (len(peak_data) - 1) and (peak_data[next_ind][1] - peak_data[curr_ind][1]) <= 23 * fps_adj:   # check that we are not out of bounds
                if not ind_exists(pks_set, ind=peak_data[next_ind][0]) and 4 * fps_adj <= peak_data[next_ind][1] - \
                        peak_data[curr_ind][1] and verify_good_peak(peak_data, next_ind, fps=fps):
                    pks_set.append(peak_data[next_ind][0])  # add the reference ind pf the peak in the peak_data main table
                next_ind += 1
    good_peak_sets = [x for x in good_peak_sets if len(x) >= 3]  # remove the sets w fewer than 3 peaks
    return good_peak_sets, peak_data


""" Add information about the displacement of the worm to the pak_data table
12th column: moving distance of worm compared to the location where the worm started scrunching (left valley of the first peak)
"""
# todo: what about sets with 3+ peaks. # issue: if the peak is in multiple sets then we will overwrite the com ultiple times

def add_com_info(good_peak_sets, com_arr, peak_data, pix_adj=1.5, mode="multiple", input_mode_times=False):
    curr_disp = 0
    if mode == "single":  #
        good_peak_sets = get_combinations(good_peak_sets, mode="any")
    good_peak_sets = check_not_too_far(good_peak_sets, peak_data)  # remove sets that are too far away
    if len(good_peak_sets)==0:
        print("peaks are too far away -> will be removed")
        peak_data = 0
        return peak_data
    for sset_ind, sset in enumerate(good_peak_sets):
        times_set = to_timestamps(good_peak_sets[sset_ind], peak_data)
        first_peak_ind = sset[0]
        start_moving_time = peak_data[first_peak_ind][1] - peak_data[first_peak_ind][6]
        start_com = com_arr[start_moving_time]
        while np.isnan(start_com[0]):  # if there is no record of com for that timestamp, get the next available com
            start_moving_time += 1
            start_com = com_arr[start_moving_time]
        for i, peak_ind in enumerate(sset):
            curr_time = times_set[i]
            com = com_arr[curr_time]
            peak_data[sset[i]][12] = com
            while np.isnan(com[0]) and curr_time < len(com_arr) - 1:  # check that we are within the bounds
                curr_time += 1
                start_com = com_arr[curr_time]
            prev_disp = curr_disp
            curr_disp = np.linalg.norm(
                np.array(start_com - np.array(com)))  # displacement from where the worm started scrunching
            peak_data[sset[i]][12] = com
            print("curr disp", curr_disp,"prev disp", prev_disp)

            if 4 > i > 0 and (prev_disp > curr_disp) or (
                    curr_disp - prev_disp) > 29 * 2 * pix_adj:  # if the worm moves >29 pix*pix_adj between 2 peaks
                good_peak_sets.remove(sset)  # remove this peak set
                print("removing due to bad COMs")
                break
            else:
                peak_data[peak_ind][11][sset_ind] = curr_disp
    #print(len(good_peak_sets), "peak sets after checking displacements")
    if len(good_peak_sets)==0:
        peak_data=0
    return peak_data



def add_com_info_new(good_peak_sets, com_arr, peak_data, pix_adj=1.5, mode="multiple", input_mode_times=False):
    if mode == "single":  #
        good_peak_sets = get_combinations(good_peak_sets, mode="any")
    good_peak_sets = check_not_too_far(good_peak_sets, peak_data)  # remove sets that are too far away
        #good_peak_sets=[good_peak_sets]
    if len(good_peak_sets)==0:
        print("peaks are too far away -> will be removed")
        peak_data = 0
    for sset_ind in range(len(good_peak_sets)):
        if not input_mode_times:
            times_set = to_timestamps(good_peak_sets[sset_ind], peak_data)
            sset = np.arange(0, len(good_peak_sets[sset_ind]), 1)
        else:
            times_set = good_peak_sets[sset_ind]
            sset = np.arange(0, len(good_peak_sets[sset_ind]), 1)
        for i in range(len(times_set)):
            curr_com = com_arr[times_set[i]]
            peak_data[sset[i]][12] = curr_com
            if i>0:
                if len(good_peak_sets) == 0:
                    break
                curr_disp = np.linalg.norm(np.array(com_arr[times_set[i-1]] - np.array(curr_com)))
                if curr_disp > 40 * pix_adj or curr_disp < 3 * pix_adj:
                    good_peak_sets = [a for a, skip in zip(good_peak_sets, [np.allclose(a, sset) for a in good_peak_sets]) if not skip] # remove this peak set
                    #good_peak_sets.remove(sset)  # remove this peak set
                    #print(com_arr[times_set[i-1]], np.array(curr_com))
                    print("worm moved too much/too little =", curr_disp, "~frame ", times_set[i], "-removing this set" )
                    #sset_ind += 1
                    continue
    if mode == "single" and len(good_peak_sets) == 0:
        peak_data=0
    return peak_data


""" Remove peak sets with high (>0.6 proportion of noise peaks) 
good_peak_sets: a list of list containing informaton about all peak sets
all_peak: a list of all peaks identified prior to appying any filter
"""
def remove_noisy_sets(good_peak_sets, all_peaks, peak_data):
    for set_ind, sset in enumerate(good_peak_sets):
        times = to_timestamps(sset, peak_data)
        all_peaks_sset = all_peaks.index(times[-1]) - all_peaks.index(times[0]) - len(sset)   # number of noise peaks in between the first and last peaks of the set
        if all_peaks_sset/len(sset) > 0.6:  # if the proportion of noise peaks is >0.6 (proportion of good peaks <0.4)
            good_peak_sets.remove(sset)
            print("removing set", set_ind, "- noisy")
    return list(good_peak_sets)


""" Check the following scrunching criteria: #
2) elongation takes more time than contraction. A fraction of peaks in the oscillation should have speed of elongation > speed of contraction
4): no elongation took more than 14 frams --> distance of peak with its previous valley always <14
"""
def check_faster_contraction(sset, peak_data, fps=5):
    # distance to the left valley (peak_data[][6]) == elongation (for local MAX)
    # distance to the right valley (peak_data[][7]) == contraction  (for local MAX)
    faster_elong_count = 0
    fps_adj = fps / 5
    ssets = get_combinations(sset)
    #print("ssets", ssets)
    for sset in ssets:
        for peak_ind in sset:
            if peak_data[peak_ind][6] > 14 * fps_adj * 2:
                #print("elongation is too long (=", peak_data[peak_ind][6], ") for set", to_timestamps(sset, peak_data))
                #if len(ssets)>1:
                #    print("Checking the next sset (out of", len(ssets), ")")
                continue
            elif peak_data[peak_ind][6] > peak_data[peak_ind][7]:  # faster elongation means that there are fewer
                faster_elong_count += 1
    if faster_elong_count == 0:
        print("no peaks where elongation is longer than contracton for", to_timestamps(sset, peak_data))
        return False
    else:
        return True


""" Check rules 3: 
 - mean of contraction amplitude > 7 pixels --> difference of peak value with its NEXT valley value > 7  
 - no contraction amplitude > 38 pixels
 - std of the amplitude of these contraction < 10 """
# sset is a list of INDEXES in the peak_data table
def check_good_amplitudes(sset, peak_data, pix_adj=1.5, printout=False, sset_mode="any"):
    worm_size_adj = 1  # todo
    ssets = get_combinations(sset, mode=sset_mode)  # get all permutations
    ssets = check_not_too_far(ssets, peak_data) # remove sets that are too far away
    #print("ssets after checking distances")
    #for sset in ssets:
    #    print(to_timestamps(sset, peak_data))
    filtered_ssets = []
    for sset in ssets:
        contr_amplitudes = []
        for peak_ind in sset:
            curr_amplitude = peak_data[peak_ind][10]  #
            if curr_amplitude < 5*pix_adj*worm_size_adj:  # no amplitude should be too low
                #if printout:
                #    print("amplitude is too low (", curr_amplitude, ") for", to_timestamps(sset,peak_data))
                if len(ssets)==1:
                    return False
                else:  # if there are other permutations left to check
                    continue
            if curr_amplitude < (38 * pix_adj * worm_size_adj):  # todo: worm size adj
                contr_amplitudes.append(curr_amplitude)
                if len(contr_amplitudes) >= 3 and mean(contr_amplitudes) > 7 * pix_adj and stdev(
                        contr_amplitudes) < 10*pix_adj:  # if any combination of 3 peaks meets the requirements
                    # todo: how do we scale stdev?
                    filtered_ssets.append(sset)
            elif len(ssets) > 1:
                #if printout:
                #    print("curr amplitude is", curr_amplitude, "pix. Checking next sset")
                continue
    if len(filtered_ssets) >= 1:
        return filtered_ssets
    else:
        return False


def check_good_widths(sset, peak_data, pix_adj=1.5, sset_mode="any"):
    worm_size_adj = 1  # todo
    ssets = get_combinations(sset, mode=sset_mode)  # get all permutations
    ssets = check_not_too_far(ssets, peak_data) # remove sets that are too far away
    for sset in ssets:
        curr_widths = []
        for peak_ind in sset:
            curr_widths.append(peak_data[peak_ind][4])
        if mean(curr_widths) > 3 and stdev(curr_widths)<3:
            print("STDEV", stdev(curr_widths))
            return True
        else:
            continue
    print(peak_data[0][1])
    print("mean wIDTH ", mean(curr_widths), "stdv", stdev(curr_widths))
    return False





# checks rule 7) (the fraction peaks with distance to next valley > 10) < 0.15
def good_valley_dists_frac(sset, peak_data, fps=5):
    count = 0
    fps_adj = fps / 5
    for peak_ind in sset:
        if peak_data[peak_ind][7] > 10 * fps_adj:
            count += 1
    if count / len(sset) < 0.5:
        # if count / len(sset) < 0.5:  todo check -- arbitrary increased from 0.15 bc doesn't make sense??
        return True
    else:
        return False


""" In the original script one of the rules was that the mean of worm aspect ratio during oscillation sould be  > 6. 
However, when i tried this, this was filtering out essetially all peak sets, so I lowered the cutoff from 6 to 2 
    asp_ratio_arr is an array containing aspect ratios for each frame of the movie (calculated in a separate script)
"""
def good_aspect_ratio(asp_ratio_arr, sset, peak_data):
    # times the worm started (left base of the 1st peak) and ended scrunching (right base of the last peak)
    start, end = [peak_data[sset[0]][1] - peak_data[sset[0]][6], peak_data[sset[-1]][1] + peak_data[sset[-1]][7]]
    ratios = asp_ratio_arr[start:end + 1]
    ratios = [x for x in ratios if x != 1.0]  # remove all asp_ratios ==1
    if np.nanmean(ratios) > 3:  # used to be >6
        return True
    else:
        print("asp ratios are bad", mean(ratios))
        return False


"""
   For one scrunching oscillation (scrunched >3 times), it should meet these criteria:
     1) the scrunching oscillation is usually clean. Thus # of noise peak (peaks didn't meet the entire criteria applied above)/# of main oscialltion peak < 0.6
     2) elongation takes more time than contraction. A fraction of peaks in the oscillation should have speed of elongation > speed of contraction
     3) mean of contraction amplitude > 7 pixels --> difference of peak value with its previous valley value > 7
        && no contraction amplitude > 38 pixels
        && std of the amplitude of these contraction < 10
     4) no elongation took more than 14 frams --> distance of peak with its previous valley always <14
     5) mean of peak width in the oscillation > 3
     6) mean of peak prominence > 7
     7) (the fraction peaks with distance to next valley > 10) < 0.15
     8) worm always move forward during scrunching. Thus, after each contraction, worm position to the location where it started its first scrunching should incrase.
        && worm can not move more than 29 pixels during one scrunching
     9) mean of worm aspect ratio (length^2/area) during the oscillation  > 6  (usually 8~13 for normally glidign worm).
"""
def analyze_peak_sets(good_peak_sets, peak_data, all_peaks, asp_ratio_arr, fps=5, pix_adj=1.5):
    fps_adj = fps / 5
    new_good_peaks = []
    good_peak_sets = remove_noisy_sets(good_peak_sets, all_peaks, peak_data)  # 1) filter out sets with a lot of noise
    for ind, sset in enumerate(good_peak_sets):
        if not check_faster_contraction(sset, peak_data, fps):  # 2) speed of elongation > speed of contraction  4) check that no elongation > 14 frames
            print("removing", ind, "by rules 2/4")
            continue
        elif not check_good_amplitudes(sset, peak_data, pix_adj):  # 3) "correct" amplitudes
            print("removing", ind, "by rule 3 (amplitudes)")
            continue
        elif mean([peak_data[i][4] for i in sset]) < 3 * fps_adj:  # 5) mean peak width in the oscillation > 3
            print("removing", ind, "by rule 5")
            continue
        elif mean([peak_data[i][5] for i in sset]) > 67 * pix_adj:  # 6) mean of peak prominence > 7
            print("removing", ind, "by rule 6")
            continue
        elif not good_valley_dists_frac(sset, peak_data, fps):  # 7) (the fraction peaks with distance to next valley > 10) < 0.15
            print("removing", ind, "by rule 7")
            continue
        # 8) moving distance; <29 pix in one scrunching oscillation
            # removed during the initial sorting
        # 9) check aspect ratios
        elif not good_aspect_ratio(asp_ratio_arr, sset, peak_data):
            print("removing", ind, "by rule 9 (aspect ratio)")
            continue
        else:
            new_good_peaks.append(sset)
    return new_good_peaks


def check_early_peaks(good_peak_sets_final, peak_data):
    count_early = 0
    for sset in good_peak_sets_final:
        sset = to_timestamps(sset, peak_data)
        if any(x < 350 for x in sset):
            print("well", ind, "has peaks earlier than 350 frame")
            count_early +=1
    if count_early>0:
        return True, count_early
    else:
        return False, count_early


def check_elongation_len(sset, peak_data, currMAL):
    ssets = get_combinations(sset)  # get all permutations
    ssets = check_not_too_far(ssets, peak_data) # remove sets that are too far away
    for sset in ssets:
        times = to_timestamps(sset, peak_data)
        worm_size = mean(currMAL[times[0]-100:times[-1]+100])
        # check that every elong len in a sset is good
        MALs_arr = [currMAL[i] for i in times]
        if all(MALs_arr > worm_size*0.5):
            print("good sset", times)
            return True
        else:
            print("bad sset", times, "mean MAL", mean(MALs_arr))
            continue
    return False


# lag -- left valley of the first peak in a set
# currMAL == smoothed MAL
def get_peak_data_set(lag, smoothedMAL, signal_unpadded):
    leeway = 10
    frame = [lag - leeway, lag + len(signal_unpadded) + leeway]
    currMAL = smoothedMAL[frame[0]:frame[1]]
    peakinds, peak_dict = find_good_peaks(currMAL, show=False, outpath=None, fps=5, pix_adj=1.5)

    v_frame = [lag - 2*leeway, lag + len(signal_unpadded) + 2*leeway]
    vinds, _ = find_good_peaks(-smoothedMAL[v_frame[0]:v_frame[1]], show=False, outpath=None, fps=5, pix_adj=1.5)
    # adjust the vinds to account for differences in frame
    vinds = [ind-leeway for ind in vinds]

    peak_data = np.zeros([len(peakinds), 14], dtype=object)  # initialize the array to store peak data
    for i in range(len(peakinds)):

        # 1: length of the peak -- here index in this data table (for future reference)
        peak_data[i][0] = int(i)
        # 2: location (time) of the peak (in frames)
        peak_data[i][1] = int(peakinds[i]) + lag - leeway
        left_valley, right_valley = match_peaks_valleys(peakinds[i], vinds)
        left_valley, right_valley = left_valley+lag-leeway, right_valley+lag-leeway

        peak_data[i][2] = right_valley  # 3rd column: CHANGED -- RIGHT VALLEY IND
        # 5th column: width of the peak
        peak_data[i][4] = peak_dict['widths'][i]
        # 6th column: prominence of the peak
        peak_data[i][5] = peak_dict['prominences'][i]
        peak_data[i][6] = peak_data[i][1] - left_valley # 7th: distance to the previous valley (in frames)
        peak_data[i][7] = right_valley - peak_data[i][1] # 8th: distance to the next valley (in frames)
        peak_data[i][8] = smoothedMAL[left_valley]          # 9th column: value of the previous valley
        peak_data[i][9] = peak_data[i][1]-peak_data[i][8]  # 10th column: diff of values of peak and its previous valley
        peak_data[i][10] = currMAL[peakinds[i]] - smoothedMAL[right_valley] #11th column: difference of values of peak and its next valley
        """ 
        peak_data[i][2] = peak_dict['left_bases'][i]
        peak_data[i][6] = peakinds[i] - peak_dict['left_bases'][i]  # 7th: distance to the previous valley (in frames)
        peak_data[i][7] = peak_dict['right_bases'][i] - peakinds[i]  # 8th: distance to the next valley (in frames)
        # 9th column: value of the previous valley
        peak_data[i][8] = currMAL[peak_dict['left_bases'][i]]
        # 10th column: difference of values of peak and its previous valley
        peak_data[i][9] = currMAL[peakinds[i]] - peak_data[i][8]
        # !! CHANGED !! 11th column: difference of values of peak and its next valley
        peak_data[i][10] = currMAL[peakinds[i]] - currMAL[peak_dict['right_bases'][i]]
        """
        # 12th: dict where key: set index, value:COM when the worms first started moving
        peak_data[i][11] = {}
        # 13th: com
        peak_data[i][12] = []  #pl
        # 14th:
        #peak_data[i][13] = currMAL[peak_dict['right_bases'][i]]
        peak_data[i][13] = smoothedMAL[right_valley]

    peakinds_new = peakinds + lag - leeway
    peak_data_new = peak_data
    return peakinds_new, peak_data_new




def zoom(zoom_sset, peak_data, smoothedMAL, good_peak_sets_final=None, mode="times"):
    if not mode == "times":
        zoom_sset = to_timestamps(zoom_sset, peak_data)
    plt.plot(smoothedMAL)
    if good_peak_sets_final:
        for sset in good_peak_sets_final:   #plot the sets that are actually were classified as scrunching
            sset = to_timestamps(sset, peak_data)
            plt.plot(sset, np.array(smoothedMAL)[sset], marker="X", markersize=4)
    # plot the peaks of interest
    plt.plot(zoom_sset, np.array(smoothedMAL)[zoom_sset], marker="o", markersize=6)
    plt.xlabel('time, frames')
    plt.ylabel('MAL, pix')
    plt.xlim(zoom_sset[0]-20, zoom_sset[-1]+20)
    plt.close(0)

""" 
fps = 5
fps_adj = fps / 5
pix_adj = 1.5

MALs, COMs, AspRatios, total = [], [], [], []

has_early_peaks = {}
early_peak_count = 0   # number of sets w
all_sets_count = 0 # number of identified peak sets

filepath = "/Volumes/Collins_Lab/15"
#filepath = "/Users/Arina/Desktop"
wells = np.arange(1, 48, 1)
wells = [1,2]

peakDataFolder = filepath + '/peak_data'
if exists(peakDataFolder) is False:
    makedirs(peakDataFolder)



for ind in wells:
    filename = filepath + "/results/well_data/MAL_well" + str(ind) + ".csv"
    currMAL = genfromtxt(filename, delimiter=',')
    currMAL = np.array(pd.DataFrame(currMAL).interpolate())
    currMAL = currMAL.reshape(-1, order='F')
    smoothing_frac = 6/len(currMAL)
    smoothedMAL = frac_lowess(currMAL, frac=smoothing_frac)  # todo this might need to be adjusted
    #MALs.append(smoothedMAL)
    MALs.append(smoothedMAL)

    filename = filepath + "/results/well_data/COM_well" + str(ind) + ".csv"
    currCOM = genfromtxt(filename, delimiter=',')
    COMs.append(currCOM)

    filename = filepath + "/results/well_data/AspRatio_well" + str(ind) + ".csv"
    currAspRatio = genfromtxt(filename, delimiter=',')
    AspRatios.append(currAspRatio)

for ind in range(len(wells)):
    all_peaks = find_all_peaks(MALs[ind])
    ## Old analysis

    peakinds, peak_dict, peak_data = get_peak_data(MALs[ind])
    good_peak_sets, peak_data = get_peak_sets(peak_data, fps=fps, pix_adj=1.5)
    print("len after get_peak_sets", len(good_peak_sets))
    peak_data = add_com_info_new(good_peak_sets, COMs[ind], peak_data, pix_adj=1.5)
    good_peak_sets_final = analyze_peak_sets(good_peak_sets, peak_data, all_peaks, AspRatios[ind], fps=5, pix_adj=1.5)
    print("len after analyze_peak_sets", len(good_peak_sets_final))
    print("FINAL: well ", wellNum, ":", good_peak_sets_final, "\n Times:",
          [to_timestamps(sset, peak_data) for sset in good_peak_sets_final])  # times of all good peaks
    outpath = os.path.expanduser(peakDataFolder + "/peak sets well" + str(wellNum) + ".png")

    #plt.plot(MALs[ind])  #smoothed
    #for sset in good_peak_sets_final:
    #    sset = to_timestamps(sset, peak_data)
    #    plt.plot(sset, np.array(MALs[ind])[sset], marker="X", markersize=4)
    #plt.xlabel('time, frames')
    #plt.ylabel('MAL, pix')
    #plt.title("Well " + str(wellNum))
    #plt.show()
    #plt.close("all")
    has_early_peaks[ind] =\
    ans, counter = check_early_peaks(good_peak_sets_final, peak_data)
    has_early_peaks[ind] = counter
    if has_early_peaks[ind]:
        early_peak_count += 1  #total number of wells w early peaks
    all_sets_count += len(good_peak_sets_final)
 
    list_of_lags_final, signal_unpadded = synth_signal.cross_correlate(MALs[ind], freq_elong=0.6, freq_contr=0.8, goal_num_sets=5, leeway=10)
    # list_of_lags_final, signal_unpadded = synth_signal.cross_correlate(MALs[ind], freq_elong=0.3, freq_contr=0.6, goal_num_sets=5, leeway=5)

    # synth_signal.generate_overlap_plots(list_of_lags_final, MALs[ind], signal_unpadded, filepath, wells[ind])

    new_peak_sets_times = []
    for lag in list_of_lags_final:
        leeway = 10
        frame = [lag-leeway, lag+len(signal_unpadded)+leeway]
        inds, _ = find_good_peaks(MALs[ind][frame[0]:frame[1]], show=False, outpath=None, fps=5, pix_adj=1.5)
        if len(inds) >= 3:
            inds += lag-leeway
            new_peak_sets_times.append(inds)

    final_peak_sets_new = []
    for lag in list_of_lags_final:
        # get peak info for one set of peaks
        peakinds_new, peak_data_new = get_peak_data_set(lag, MALs[0], signal_unpadded)
        if len(peakinds_new) < 3:
            print("fewer than 3 peaks in a set", peakinds_new)
            continue
        sset_new = np.arange(0, len(peakinds_new), 1)  # reference INDEXES in the peak_data_new table
        peak_data_new = add_com_info_new(sset_new, COMs[0], peak_data_new, pix_adj=1.5, mode="single")
        if type(peak_data_new) == bool and not peak_data_new:
            print("removing", peakinds_new, "due to bad COMs")
            continue
        #if not check_faster_contraction(sset_new, peak_data_new, fps):  # 2) speed of elongation > speed of contraction  4) check that no elongation > 14 frames
        #    print("removing", peakinds_new, "by rules 2/4")
        #    continue
        if not check_elongation_len(sset_new, peak_data_new, MALs[0]):
            print("removing", peakinds_new, "due to bad (too short) elong MAL")
            continue
        elif not check_good_amplitudes(sset_new, peak_data_new, pix_adj, printout=True):  # 3) "correct" amplitudes
            print("removing", peakinds_new, "by rule 3 (amplitudes)")
            continue
        elif mean([peak_data_new[i][4] for i in sset_new]) < 3 * fps_adj:  # 5) mean peak width in the oscillation > 3
            print("removing", peakinds_new, "by rule 5")
            continue
        elif mean([peak_data_new[i][5] for i in sset_new]) > 67 * pix_adj:  # 6) mean of peak prominence > 7
            print("removing", peakinds_new, "by rule 6")
            continue
        # 9) check aspect ratios
        elif not good_aspect_ratio(AspRatios[0], sset_new, peak_data_new):
            print("removing", peakinds_new, "by rule 9 (aspect ratio)")
            continue
        final_peak_sets_new.append(list(peakinds_new))
    total.append(final_peak_sets_new)


counter = 0
for old_set in good_peak_sets_final:
    old_set = to_timestamps(old_set, peak_data)
    for set in new_peak_sets_times:
        for new_elem in set:
            for old_elem in old_set:
                #print(old_elem, new_elem)
                if old_elem == new_elem:
                    #print("overlapping peak", old_elem)
                    counter += 1
"""







""" 
leg = []
for i, com_arr in enumerate(COMs):
    leg.append("Well " + str(wells[i]))
    velocities, displacements, disp_arr = calculate_velocities(com_arr, MALs[i], fps=fps)
    time = np.arange(start=(fps*6/2), stop=(len(com_arr)/fps-(fps*6/2)), step=6*fps)
    plt.scatter(time, displacements)
    plt.plot(time, displacements, linestyle='--')
    slope, intercept = np.polyfit(time, displacements, 1)
    plt.plot(time, time * slope + intercept, 'r')
    plt.xlabel('time, seconds')
    plt.ylabel('Mean displacement, normalized by body length')
    plt.xlim((0, (len(com_arr)/fps)))
    #plt.ylim((0, 500))
    plt.title("Well " + str(wells[i]))
    plt.legend(["Mean displacements", "y="+str(round(slope, 3))+"x+"+str(round(intercept, 3))])
    outpath = os.path.expanduser("/Users/Arina/Desktop/02/results/peak_sets/displacements well" + str(wells[i]) + ".png")
    #outpath = os.path.expanduser("/Users/Arina/Desktop/02/results/peak_sets/displacements.png")
    plt.savefig(outpath)
    plt.show()
    #plt.legend(leg)
    plt.close()
"""