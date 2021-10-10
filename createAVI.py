# import libraries
import cv2 as cv
import numpy as np


"""
Creates an AVI movies from an image sequence.
Inputs:
 - start_frame, last_frame : ints, specify the last last frames that we want to include in the movie
 - scale_percent: int, option compression factor; use 100 to keep the video at the original quality
 - fps: int, frames per second
 - filepath: filepath to the folder with the image sequnce
 - outpath: filepath to where you want to save the .avi movie
"""
def createAVI(start_frame, last_frame, scale_percent = 100, fps = 5, filepath='/Users/Desktop', outpath=None):
    if outpath==None:
        outpath = filepath + "/fullPlate.avi"
    img_array = []
    for i in range(start_frame, last_frame):
        newPath = filepath + "/" + str(i) + ".jpeg"
        print(newPath)
        img = cv.imread(newPath)
        if img is not None:
            img = cv.putText(img, str(start_frame+i), (50, 200), cv.FONT_HERSHEY_COMPLEX, 2, (200, 0, 0), 3)
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            if scale_percent != 100:
                dsize = (width, height)
                img = cv.resize(img, dsize)    # resize image
            img_array.append(img)
        else:
            continue
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    video = cv.VideoWriter(outpath, fourcc, fps, (width, height))
    for img in img_array:
        video.write(img)
    video.release()
    cv.destroyAllWindows()



# to run, uncomment the line below and change the arguments as needed
# createAVI(start_frame, last_frame, scale_percent = 100, fps = 5, filepath='/Users/Desktop', outpath=None)