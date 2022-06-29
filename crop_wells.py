import numpy as np
from os.path import exists
from os import makedirs
import read_input as rin
import filtering
import math
import crop_interface as interface
import os
import cv2 as cv
import matplotlib.pyplot as plt





# well size params are set for a 5496 × 3672 image
# if wellNum parameter is not specified or set to -1, all wells will be cropped

def crop_one(plateFolder, start_frame, end_frame, x_start, y_start, well_width, well_height, x_end=None, y_end=None, x_adj=0, y_adj=0, wells=-1, image_type=".png"):
    if type(wells) is not list:
        wells = [wells]
    wells = list(wells)
    outputPath = plateFolder + "/results"
    if os.path.exists(outputPath) is False:  
        # create a folder where the script output would be stored 
        # if it does not yet exists 
        os.makedirs(outputPath)


    # plateFolder = "/Users/Arina/Desktop/02" + str(i + 1)  #TODO change later when we want to look at more plates
    # wellFolder = './well_data/input/' + output_name + '/well_' + str(i+1)
    if exists(plateFolder) is False:
        makedirs(plateFolder)
    for j in range(48):
        wellFolder = outputPath + '/well_' + str(j + 1)
        if exists(wellFolder) is False:
            makedirs(wellFolder)

    # filepath = filepath + dataset_name + "/"
    # Need to establish well plate naming convention and change the filepath here
    imgs = rin.read_input(start_frame=start_frame,
                          end_frame=end_frame, filepath=plateFolder)
    print("start cropping")
    # for frame, img in enumerate(imgs[plate - 1]):
    for frame, img in enumerate(imgs):
        if img is not None:
            counter = 0
            # for i in range(y_start, y_end, well_width):
            #    for j in range(x_start, x_end, well_width):
            for i in range(y_start, y_start+6*(well_height+y_adj), well_height+y_adj):
                for j in range(x_start, x_start+(8*well_width+x_adj), well_width+x_adj):
                    counter += 1
                    print("\n\ntype", type(wells))
                    print(wells)
                    if (-1 in wells) or (counter in wells):
                        x_1 = x_start + (j - x_start)
                        y_1 = y_start + (i - y_start)
                        y_2 = y_1 + well_height
                        x_2 = x_1 + well_height
                        cropped = img[y_1:y_2,  x_1:x_2]
                        impath = outputPath + "/" + "well_" + \
                            str(counter) + "/" + "croppedImage" + \
                            "_" + str(start_frame + frame) + image_type
                        cv.imwrite(impath, cropped)
                        #print("writing", impath)
    del imgs  # explicitly free up memory


def crop(x_start=250, y_start=35, x_end=1840, y_end=1170, well_width=190, well_height=190, start_frame=1, end_frame=100):
    # Select the corner wells. Press Q to quit. Press R to restart.
    print("Please select the corner wells")
    refPts = interface.getPoints(plateFolder)
    x_start, y_start = int(
        refPts[0][0]-0.5*well_width), int(refPts[0][1]-0.5*well_height)
    x_end, y_end = int(
        refPts[1][0]+0.5*well_width), int(refPts[2][1]+0.5*well_height)
    x_adj = math.ceil((refPts[4][0] - refPts[1][0])/8)  # (x_4-x_2)/8
    y_adj = math.ceil((refPts[1][1] - refPts[0][1])/8)  # (y_2-y_1)/6

    for i in range(15):  # TODO: change that!!!!! 15 is bc we process 100 at once
        crop_one(plateFolder=plateFolder, start_frame=start_frame,      end_frame=end_frame, x_start=x_start, y_start=y_start, x_adj=x_adj, y_adj=y_adj, well_width=well_width, well_height=well_height, wells=wells, image_type=".png")
        start_frame = end_frame
        end_frame = end_frame + 100


if __name__ == "__main__":

    ####### SPECIFY THE PLATE FOLDER ######
    plateFolder = '/Users/arina/Desktop/Neuro98 articles + misc/2021_08_12 Arina 3 chem scrunching/18'
    # change these if necessary
    image_type = ".png"
    outputPath = plateFolder + "/results"

    ###### CHANGE THIS IF YOU WANT TO CROP ONLY SPECIFIC WELLS ####
    # e.g. To crop wells 1 and 2 only, replace the line below with 
    # wells = [1,2]
    wells = np.arange(1, 49, 1).tolist()

    start_frame = 1  
    end_frame = 100
    # end_frame - start_frame gives us the number of frames 
    # to be loaded AT ONCE for internal processing.
    # 100-150 is optimal for speed (more frames is too memory intense)

    # Well cropping params
    # you can try to adjust these if the plate is severely misalligned 
    x_start = 250
    y_start = 35
    x_end = 1840
    y_end = 1170
    well_width = 190
    well_height = 190
    crop(x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end,
         well_width=well_width, well_height=well_height)
