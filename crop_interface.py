import cv2
import numpy as np
import argparse
import argparse


# You then need to bind that function to a window that will capture the mouse click
""" 
newPath = "/Volumes/Collins_Lab/15/1.jpeg"
img = cv2.imread(newPath)
cv2.namedWindow('image')

"""
refPt = []
def click_and_crop(event, x, y, flags, img):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        # draw a rectangle around the region of interest
        # draw a rectangle around the region of interest
        cv2.circle(img, refPt[-1], 93, (0, 255, 0), 2)
        cv2.imshow("image", img)



def getPoints(plateFolder):
    global refPt
    imgPath = plateFolder + "/1.jpeg"
    img = cv2.imread(imgPath)
    clone = img.copy()
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_and_crop, img)
    # keep looping until the 'q' key is pressed
    while (len(refPt)) <= 4:
        # display the image and wait for a keypress
        #cv2.imshow("Select the corner wells. Press Q to quit. Press R to restart.", img)
        cv2.imshow("image", img)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
        elif key == ord("r"):
            refPt=[]
            img=clone

    # refPt[0][0] - 1/2 * well_width
    # refPt[0][1] - 1/2 * well_height
    #
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return refPt


#getPoints(plateFolder= "/Volumes/Collins_Lab/15")
""" 
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Path to the image")
    args = vars(ap.parse_args())
    imgPath = args["image"] if args["image"] else 0
    # now get our image from either the file or built-in webcam
    image = cv2.imread(imgPath)
    if image is not None:
        cv2.namedWindow('CapturedImage', cv2.WINDOW_NORMAL)
        cv2.imshow('CapturedImage', image)
        # specify the callback function to be called when the user
        # clicks/drags in the 'CapturedImage' window
        cv2.setMouseCallback("image", click_and_crop)
        while True:
            # wait for Esc or q key and then exit
            krey = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print('Image cropped at coordinates: {}'.format(refPt))
                cv2.destroyAllWindows()
                break
"""
