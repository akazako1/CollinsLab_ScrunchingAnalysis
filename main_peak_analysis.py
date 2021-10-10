
import numpy as np
import os
import pandas as pd
from numpy import genfromtxt
from os import makedirs
from os.path import exists
from statistics import stdev, mean
import statsmodels.api as sm
import peak_analysis as pa



fps = 5
fps_adj = fps / 5
pix_adj = 1.5

MALs, COMs, AspRatios, total = [], [], [], []


has_early_peaks = {}
early_peak_count = 0   # number of sets w
all_sets_count = 0  # number of identified peak sets

filepath = "/Volumes/Collins_Lab/15"
plateFolder = "/Volumes/Collins_Lab/My data/Christinas plates/2021_08_12 Arina 3 chem scrunching/16"

wells = np.arange(1, 49, 1)
wells = [1]
peakDataFolder = filepath + '/peak_data'
if exists(peakDataFolder) is False:
    makedirs(peakDataFolder)


for well in wells:   # for every well in the list of wells

    filename = filepath + "/results/well_data/MAL_well" + str(well) + ".csv"
    currMAL = genfromtxt(filename, delimiter=',')
    currMAL = np.array(pd.DataFrame(currMAL).interpolate())
    currMAL = currMAL.reshape(-1, order='F')
    smoothing_frac = 6/len(currMAL)
    smoothedMAL = pa.frac_lowess(currMAL, frac=smoothing_frac)  # todo this might need to be adjusted
    MALs.append(smoothedMAL)

    filename = filepath + "/results/well_data/COM_well" + str(well) + ".csv"
    currCOM = genfromtxt(filename, delimiter=',')
    COMs.append(currCOM)

    filename = filepath + "/results/well_data/AspRatio_well" + str(well) + ".csv"
    currAspRatio = genfromtxt(filename, delimiter=',')
    AspRatios.append(currAspRatio)



all_peak_data = []
for ind in range(len(wells)):
    all_peaks = pa.find_all_peaks(MALs[ind])

    list_of_lags_final, signal_unpadded = pa.synth_signal.cross_correlate(MALs[ind], freq_elong=0.6, freq_contr=0.8, goal_num_sets=5, leeway=10)
    #pa.synth_signal.generate_overlap_plots(list_of_lags_final, MALs[ind], signal_unpadded, filepath, wells[ind])

    new_peak_sets_times = []
    for lag in list_of_lags_final:
        leeway = 10
        frame = [lag-leeway, lag+len(signal_unpadded)+leeway]
        inds, _ = pa.find_good_peaks(MALs[ind][frame[0]:frame[1]], show=False, outpath=None, fps=5, pix_adj=1.5)
        if len(inds) >= 3:
            inds += lag-leeway
            new_peak_sets_times.append(inds)


    final_peak_sets_new = []
    for lag in list_of_lags_final:
        # get peak info for one set of peaks
        peakinds_new, peak_data_new = pa.get_peak_data_set(lag, MALs[ind], signal_unpadded)
        if len(peakinds_new) < 3:
            print("fewer than 3 peaks in a set", peakinds_new)
            continue
        sset_new = np.arange(0, len(peakinds_new), 1)  # reference INDEXES in the peak_data_new table
        peak_data_new = pa.add_com_info(sset_new, COMs[ind], peak_data_new, pix_adj=1.5, mode="single")
        all_peak_data.append(peak_data_new)
        if type(peak_data_new)==int and peak_data_new==0:  #add_com_info_new returns 0 when COMs are bad
            print("removing", peakinds_new, "due to bad COMs")
            continue
        # if not check_faster_contraction(sset_new, peak_data_new, fps):  # 2) speed of elongation > speed of contraction  4) check that no elongation > 14 frames
        #    print("removing", peakinds_new, "by rules 2/4")
        #    continue
        if not pa.check_elongation_len(sset_new, peak_data_new, MALs[ind]):
            print("removing", peakinds_new, "due to bad (too short) elong MAL")
            continue
        elif not pa.check_good_amplitudes(sset_new, peak_data_new, pix_adj, printout=True):  # 3) "correct" amplitudes
            print("removing", peakinds_new, "by rule 3 (amplitudes)")
            continue
        elif not pa.check_good_widths(sset_new, peak_data_new, pix_adj=1.5, sset_mode="any"):
        #elif mean([peak_data_new[i][4] for i in sset_new]) < 3 * fps_adj:  # 5) mean peak width in the oscillation > 3
            print("removing", peakinds_new, "by rule 5")
            continue
        elif mean([peak_data_new[i][5] for i in sset_new]) > 67 * pix_adj:  # 6) mean of peak prominence > 7
            print("removing", peakinds_new, "by rule 6")
            continue
        # 9) check aspect ratios
        elif not pa.good_aspect_ratio(AspRatios[ind], sset_new, peak_data_new):
            print("removing", peakinds_new, "by rule 9 (aspect ratio)")
            continue
        else:
            final_peak_sets_new.append(list(peakinds_new))

        #pa.zoom(peakinds_new, peak_data_new, smoothedMAL,mode="times")

    print("WELL", wells[ind],":",final_peak_sets_new)
    total.append(final_peak_sets_new)


count = 0
resPy = []
# print all inds of wells that scrunch
for ind in range(len(total)):
    if len(total[ind])>0:
        #print(ind)
        count += 1
        resPy.append(1)
    else:
        resPy.append(0)
print("total scrunching", count)


resMAT = [1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,1,1,1,1,1,0,1]
corr_count = 0
falseNegPy = []
falsePosPy = []
for i, ans in enumerate(resPy):   # check every response generated by Python script
    if resPy[i] == resMAT[i]:
        corr_count += 1
    elif resPy[i]==0:
        falseNegPy.append(i)
    elif resPy[i]==1:
        falsePosPy.append(i)  # list of inds of false positives produced by Python script. ind == wellNum - 1

# for plate 15
for ind in falseNegPy:
    print("well", ind+1, "is classified as 0 by Python script by not MATLAB" )
for ind in falsePosPy:
    print("well",ind+1, "is classified as 1 by Py script. Good peaks:", total[ind])


