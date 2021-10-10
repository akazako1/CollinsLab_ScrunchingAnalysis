import os
import cv2 as cv
import read_input as rin
from os.path import exists
from os import makedirs



""" This is the main image cropping function. 
'wells' is an optional argument specifying the wells to be cropped.
If 'wells' is is not specified or set to '-1', 
all wells will be cropped and saved.
"""
# if wellNum parameter is not specified or set to -1, all wells will be cropped
def crop(num_plates, wells =-1, start_frame=0, end_frame=500, x_start=350, y_start=75, x_end=4900, y_end=3200, well_width=590, well_height=590,
         outputPath="/Desktop/results", plateFolder = "/Desktop/plateFolder", image_type=".png"):
    if type(wells) is not list:
        wells = [wells]
    wells = set(wells)

    if os.path.exists(outputPath) is False:
       os.makedirs(outputPath)

    for i in range(num_plates):
        if exists(plateFolder) is False:
            makedirs(plateFolder)
        for j in range(48):
            wellFolder = outputPath + '/well_' + str(j + 1)
            if exists(wellFolder) is False:
                makedirs(wellFolder)

    imgs = rin.read_input(num_plates, start_frame=start_frame, end_frame=end_frame, filepath=plateFolder)     # Need to establish well plate naming convention and change the filepath here
    plate = 1

    for i in range(num_plates):
        plate += 1
        #for frame, img in enumerate(imgs[plate - 1]):
        for frame, img in enumerate(imgs):
            if img is not None:
                counter = 0
                for i in range(y_start, y_start+6*well_height, well_height):
                    print("y coord", i)
                    for j in range(x_start, x_start+8*well_width, well_width):
                            counter += 1
                            if (-1 in wells) or (counter in wells):
                                x_1 = x_start + (j - x_start)
                                y_1 = y_start + (i - y_start)
                                y_2 = y_1 + well_height
                                x_2 = x_1 + well_height
                                cropped = img[y_1:y_2,  x_1:x_2]
                                impath = outputPath + "/" + "well_" + str(counter) + "/" + "croppedImage" + "_" + str(start_frame + frame + 1) + image_type
                                cv.imwrite(impath, cropped)
                                print("writing", impath)
    del imgs  # explicitly free up memory


