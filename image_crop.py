import os
import cv2 as cv
import numpy as np
import read_input as rin
import matplotlib.pyplot as plt
from os.path import exists
from os import makedirs


# well size params are set for a 5496 × 3672 image
# if wellNum parameter is not specified or set to -1, all wells will be cropped
def crop(plateFolder, start_frame, end_frame, x_start, y_start, well_width, well_height, x_end=None, y_end=None, x_adj=0, y_adj=0,num_plates=1, wells =-1, image_type=".png"):
    if type(wells) is not list:
        wells = [wells]
    wells = set(wells)
    outputPath = plateFolder + "/results"
    if os.path.exists(outputPath) is False:
       os.makedirs(outputPath)

    for i in range(num_plates):
        # plateFolder = "/Users/Arina/Desktop/02" + str(i + 1)  #TODO change later when we want to look at more plates
        # wellFolder = './well_data/input/' + output_name + '/well_' + str(i+1)
        if exists(plateFolder) is False:
            makedirs(plateFolder)
        for j in range(48):
            # wellFolder = './well_data/input/' + output_name + '/well_' + str(i+1)
            wellFolder = outputPath + '/well_' + str(j + 1)
            if exists(wellFolder) is False:
                makedirs(wellFolder)

        # filepath = filepath + dataset_name + "/"
    imgs = rin.read_input(num_plates, start_frame=start_frame, end_frame=end_frame, filepath=plateFolder)     # Need to establish well plate naming convention and change the filepath here
    plate = 1
    print("start cropping")

    for i in range(num_plates):
        plate += 1
        #for frame, img in enumerate(imgs[plate - 1]):
        for frame, img in enumerate(imgs):
            if img is not None:
                counter = 0
                #for i in range(y_start, y_end, well_width):
                #    for j in range(x_start, x_end, well_width):
                for i in range(y_start, y_start+6*(well_height+y_adj), well_height+y_adj):
                    for j in range(x_start, x_start+(8*well_width+x_adj), well_width+x_adj):
                            counter += 1
                            if (-1 in wells) or (counter in wells):
                                x_1 = x_start + (j - x_start)
                                y_1 = y_start + (i - y_start)
                                y_2 = y_1 + well_height
                                x_2 = x_1 + well_height
                                cropped = img[y_1:y_2,  x_1:x_2]
                                impath = outputPath + "/" + "well_" + str(counter) + "/" + "croppedImage" + "_" + str(start_frame + frame) + image_type
                                cv.imwrite(impath, cropped)
                                #print("writing", impath)
    del imgs  # explicitly free up memory


