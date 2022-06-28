"""
This is the main scrunching tracking script
It assumes that the raw data has been preprocessed: individuals wells cropped
and the images of individual wells were saved in the corresponding folders



Output: the following data is generated for each of the respective wells
1) MAL (txt), 2) MAL vs time plot, 3) COM (txt), Aspect Ratio (txt)

"""
import numpy as np
from numpy import asarray
from numpy import savetxt
import csv
import matplotlib
import visualize_results
#matplotlib.use('Qt5Agg')  # Apple doesn't like Tkinter (TkAgg backend) so I needed to change the backend to 'Qt5Agg'
from matplotlib import pyplot as plt
from scipy import signal
import cv2 as cv
import os
from os.path import exists
from os import makedirs
import scrunching_track

plateFolder = "/Users/arina/Desktop/Neuro98 articles + misc/2021_08_12 Arina 3 chem scrunching/17"
outputPath = plateFolder + "/results"
wellDataFolder = outputPath + '/well_data'

wells = list(np.arange(2, 49, 1))
#wells = [16, 10, 22, 40, 48]
wells = [3, 41, 48]     

start_frame=1
end_frame=1500


centermost_arr, mals_arr, coms_arr, asp_ratios_arr = [], [], [], []

for ind in wells:
    curr_centermost_arr, curr_mal_arr, curr_com_arr, curr_asp_ratio_arr = \
        scrunching_track.analyze(wellNum=ind, plateFolder=plateFolder, start_frame=1, end_frame=1500)
    centermost_arr.append(curr_centermost_arr)
    mals_arr.append(curr_mal_arr)
    coms_arr.append(curr_com_arr)
    asp_ratios_arr.append(curr_asp_ratio_arr)


    if exists(outputPath) is False:
            makedirs(outputPath)

    wellDataFolder = outputPath + '/well_data'
    if exists(wellDataFolder) is False:
            makedirs(wellDataFolder)
    wellVidsFolder = outputPath + '/well_vids'
    if exists(wellVidsFolder) is False:
            makedirs(wellVidsFolder)

    #visualize_results.displayVideo(filtered_imgs=np.array(curr_centermost_arr),outpath=wellVidsFolder + "/" + "binary_well" + str(ind) + '.avi')

    visualize_results.plotMAL(curr_mal_arr, MAL=True, title=("well" + str(ind)),
                             outpath=(wellDataFolder + "/MAL plot" + str(ind)), show=False)

    # visualize_results.displayOrigVideo(start_frame=start_frame, last_frame=end_frame, filepath=plateFolder, wellNum=wellNum, outpath=wellVidsFolder + "/" + "orig_well" + str(wellNum) + '.avi')

    data = asarray(curr_mal_arr)
    path = os.path.expanduser(wellDataFolder + '/MAL_well' + str(ind) + '.csv')
    savetxt(path, data, delimiter=',', fmt='%1.3f')


    x = [tup[0] for tup in curr_com_arr]
    y = [tup[1] for tup in curr_com_arr]
    xy = np.stack((x, y))
    xy = np.transpose(xy)
    path = os.path.expanduser(wellDataFolder + '/COM_well'+ str(ind)+'.csv')
    savetxt(path, xy, delimiter=',', fmt='%1.3f')


    filename = os.path.expanduser(wellDataFolder + "/centermost_well" + str(ind) + ".csv")
    curr_centermost_arr = np.array(curr_centermost_arr)
    arrReshaped = curr_centermost_arr.reshape(curr_centermost_arr.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt(filename, arrReshaped)

    data = asarray(curr_asp_ratio_arr)
    path = os.path.expanduser(wellDataFolder + '/AspRatio_well' + str(ind) + '.csv')
    savetxt(path, data, delimiter=',', fmt='%1.3f')



    #data = asarray(tracked_areas)
    #path = os.path.expanduser('Areas well'+ str(wellNum)+'.csv')
    #savetxt(path, data, delimiter=',', fmt='%1.3f')


"""
centermost_arr = []
wellVidsFolder = outputPath + '/well_vids'

for ind in wells:
    curr_centermost_arr, _, _, curr_asp_ratio_arr = \
        scrunching_track.analyze(wellNum=ind, plateFolder=plateFolder, start_frame=1, end_frame=1500)

    #visualize_results.displayVideo(filtered_imgs=curr_centermost_arr,
    #                               outpath=wellVidsFolder + "/" + "binary_well" + str(ind) + '.avi')

    #data = asarray(curr_centermost_arr)
    #path = os.path.expanduser(wellDataFolder + '/centermost_well' + str(ind) + '.csv')
    #savetxt(path, data)

    filename = os.path.expanduser(wellDataFolder + "/centermost_well" + str(ind) + ".csv")
    curr_centermost_arr = np.array(curr_centermost_arr)
    arrReshaped = curr_centermost_arr.reshape(curr_centermost_arr.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt(filename, arrReshaped)

    data = asarray(curr_asp_ratio_arr)
    path = os.path.expanduser(wellDataFolder + '/AspRatio_well' + str(ind) + '.csv')
    savetxt(path, data, delimiter=',', fmt='%1.3f')


    #loadedArr = np.loadtxt(filename)
    # loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // arr.shape[2], arr.shape[2])



    asp_ratio_arr = []
    for centermost in curr_centermost_arr:
        if sum(sum(centermost)) != 0:
            contours, _ = cv.findContours(centermost, 1, 2)
            cnt = contours[0]
            #print(cnt)
            (x, y), (width, height), angle = cv.minAreaRect(cnt)
            aspect_ratio = min(width, height) / max(width, height)
            if aspect_ratio<1:
                aspect_ratio= 1/aspect_ratio
            asp_ratio_arr.append(aspect_ratio)
        else:
            asp_ratio_arr.append(np.nan)
        data = asarray(asp_ratio_arr)
        path = os.path.expanduser(wellDataFolder + '/AspRatio_well' + str(ind) + '.csv')
        savetxt(path, data, delimiter=',', fmt='%1.3f')
       
        """