import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
#import glob
import matplotlib.image as image


"""
Creates an imgs list containing all frames for one well
"""
def read_input(start_frame, end_frame, filepath):
    imgs = []
    for i in range(start_frame, end_frame):
        """ def read_input(num_plates, start_frame,  num_frames, filepath):
        if i <= 8:
                newPath = filepath + "/00" + str(i+1) + ".jpg"
                im = cv.imread(newPath)
                imgs.insert(i, im)
        elif i > 8 and i <= 98:
                newPath = filepath + "/0" + str(i+1) + ".jpg"
                im = cv.imread(newPath)
                imgs.insert(i, im)
        else:
                newPath = filepath + str(i+1) + ".jpg"
                im = cv.imread(newPath)
                imgs.insert(i, im)
        """
        newPath = filepath + "/" + str(i) + ".jpeg"
        print('filepath reading', newPath)
        if i%100 == 0:
            print("reading ", newPath)
        # im = image.imread(newPath)
        im = cv.imread(newPath)
        imgs.insert(i, im)
    imgs = np.array(imgs)
    return imgs


""" This function assumes that the cropped file has already been created"""
def read_input_oneWell(start_frame, num_frames, filepath, wellNum):
    imgs = []
    for i in range(start_frame, num_frames):
        newPath = filepath + "/" + "results/" + "well_" + str(wellNum) + "/croppedImage_" + str(i + 1) + ".png"
        im = cv.imread(newPath)
        imgs.insert(i, im)
    imgs = np.array(imgs, dtype=object)
    return imgs

