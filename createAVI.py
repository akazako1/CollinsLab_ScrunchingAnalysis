# import libraries
import cv2 as cv
import numpy as np

refPt = []
cropping = False

    # Notes on the name conventions:
    # Folder name should NOT contain " " or '' special characters
    # Sample good folderpath: 
    # '/Users/arina/Desktop/scrunching_movies/5/'
    # In this example, `5` is the name of the folder containing the entire scrunching movie for the plate 


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)

        cv.imshow("image", img)


def draw_rect(img, x, y, well_radius):
    p0 = int(x-(well_radius/2)), int(y-(well_radius/2))
    p1 = int(x+(well_radius/2)), int(y+(well_radius/2))
    print(p0, p1)
    """ 
    cv.putText(img, str(x) + ',' + 
                str(y), (x,y), font,
                1, (255, 0, 0), 2)
    """
    cv.rectangle(img, p0, p1, (255, 0, 0), 2)   # blue rectangle
    cv.imshow('image', img)


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:

        # displaying the coordinates in the terminal
        print(x, ' ', y)

        # displaying the coordinates on the image window
        font = cv.FONT_HERSHEY_SIMPLEX

        cv.putText(img, str(x) + ',' +
                   str(y), (x, y), font,
                   1, (255, 0, 0), 2)

        cv.imshow('image', img)

        draw_rect(img, x, y, params[0])
        cv.waitKey(0)
        return x, y

    # checking for right mouse clicks
    if event == cv.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv.putText(img, str(b) + ',' +
                   str(g) + ',' + str(r),
                   (x, y), font, 1,
                   (255, 255, 0), 2)
        cv.imshow('image', img)
        return x, y


def crop_well(filepath, x, y, well_radius):
    if outpath == None:
        outpath = filepath + "/fullPlate.avi"
    for i in range(start_frame, last_frame):
        newPath = filepath + "/" + str(i) + ".jpeg"
        # print(newPath)
        img = cv.imread(newPath)
        # Cropping an image
        cropped_image = img[y-well_radius/2:y + well_radius/2,
                            x-well_radius/2:x + well_radius/2]
        # Display cropped image
        cv.imshow("cropped", cropped_image)


"""
Creates an AVI movies from an image sequence.
Inputs:
 - start_frame, last_frame : ints, specify the last last frames that we want to include in the movie
 - scale_percent: int, option compression factor; use 100 to keep the video at the original quality
 - fps: int, frames per second
 - filepath: filepath to the folder with the image sequnce
 - outpath: filepath to where you want to save the .avi movie
"""
def createAVI(start_frame, last_frame, scale_percent=100, fps=5, well_num=None, filepath='/Users/Desktop', outpath=None):
    if outpath == None:
        outpath = filepath + "/fullPlate" + ".avi"
    img_array = []
    for i in range(start_frame, last_frame):
        if well_num != -1:
            newPath = filepath + "/" + "well_" + \
                str(well_num) + "/" + "croppedImage" + "_" + \
                str(i) + ".png"
            outpath = filepath + "/well" + str(well_num) + "frames" + str(start_frame) + "-" + str(last_frame) + ".avi"
        else:
            newPath = filepath + "/" + str(i) + ".jpeg"
        print("reading image", newPath)
        img = cv.imread(newPath)
        if img is not None:
            if well_num == -1:
                img = cv.putText(img, str(start_frame+i), (50, 200),
                                cv.FONT_HERSHEY_COMPLEX, 2, (200, 0, 0), 3)
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            if scale_percent != 100:
                dsize = (width, height)
                img = cv.resize(img, dsize)    # resize image
            img_array.append(img)
        else:
            print("img is None")
            continue
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv.VideoWriter(outpath, fourcc, fps, (width, height))
    for img in img_array:
        video.write(img)
    video.release()
    cv.destroyAllWindows()


# driver function
if __name__ == "__main__":

    start_frame = int(input("Enter the start frame: "))
    last_frame = int(input("Enter the end frame: "))
    well_num = int(input("Enter the well number: "))  #or -1 for all wells

    filepath = input("Enter the filepath : ")    # path to the folder with the images
    
    # change these if necessary 
    image_type = ".png"
    outputPath = filepath + "/results"

    if int(well_num) != -1:    # get inside the wells folder
        filepath = filepath + "/results"
    #filepath = "\'" + filepath + "\'"

    print("\nfilepath is", filepath)

    #filepath =  filepath + "/well_" + str(well_num)   
    outpath = filepath + ".avi"
    createAVI(start_frame, last_frame, scale_percent=100, fps=5,
              well_num=well_num, filepath=filepath, outpath=None)
    


