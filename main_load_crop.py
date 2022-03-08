import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from os.path import exists
from os import makedirs
import read_input as rin  
#from skan import pre
#from skimage import morphology
import random as rng
import filtering
import math
import image_crop
import crop_interface as interface


refPt = []
# General params
num_plates = 1
wells = list((np.arange(2, 49, 1)))   # all wells 
wells = [10, 20,33, 13]     


# Well cropping params
start_frame = 1
end_frame = 100  # number of frames to be loaded at once for internal processing. 100-150 is optimal for speed
x_start = 250
y_start = 35
x_end = 1840
y_end = 1170
well_width = 190
well_height = 190

image_type = ".png"
plateFolder = '/Users/arina/Downloads/2021_08_12 Arina 3 chem scrunching/18'
#plateFolder = '/Users/arina/Downloads/2021_08_12 Arina 3 chem scrunching/18/results/well_9'

outputPath = plateFolder + "/results"
#outputPath = "/Volumes/DISK_IMG/02/results"



# Select the corner wells. Press Q to quit. Press R to restart.
print("Please select the corner wells")
refPts = interface.getPoints(plateFolder)
x_start, y_start = int(refPts[0][0]-0.5*well_width), int(refPts[0][1]-0.5*well_height)
x_end, y_end = int(refPts[1][0]+0.5*well_width), int(refPts[2][1]+0.5*well_height)
x_adj = math.ceil((refPts[4][0] - refPts[1][0])/8)  # (x_4-x_2)/8
y_adj = math.ceil((refPts[1][1] - refPts[0][1])/8)  # (y_2-y_1)/6

for i in range(15):
    image_crop.crop(plateFolder=plateFolder, start_frame=start_frame, end_frame=end_frame,
                    x_start=x_start, y_start=y_start, x_adj=x_adj, y_adj=y_adj, well_width=well_width,
                    well_height=well_height, num_plates=num_plates,  wells=wells, image_type=".png")
    start_frame = end_frame
    end_frame = end_frame + 100




#/Volumes/DISK_IMG/02/results/well_2

