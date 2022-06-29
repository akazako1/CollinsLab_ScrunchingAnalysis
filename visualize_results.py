# import libraries
import cv2 as cv
import numpy as np
#matplotlib.use('Qt5Agg')    # Apple doesn't like Tkinter (TkAgg backend) so I needed to change the backend to 'Qt5Agg'
from matplotlib import pyplot as plt
import read_input as rin
import data_collection
import skimage.measure as skmeasure
import glob


def plot_mean_line(data, time):
    y_mean = [np.mean(data)] * len(time)
    plt.plot(time, y_mean, label='Mean', linestyle='--')
    plt.show()


def displayVideo(filtered_imgs, outpath):
    frameSize = filtered_imgs[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    video = cv.VideoWriter(outpath, fourcc, 10, frameSize, False)   #10 fps
    for i, img in enumerate(filtered_imgs):
        img = np.uint8(img)*250
        img = cv.putText(img, str(i), (30, 30), cv.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
        video.write(img)
    video.release()
    cv.destroyAllWindows()



""" Creates and saves a video from raw images for a particular well 
NOTE: change the output path every time -- othervise the movie would be corrupted

"""
def displayFullVideo(start_frame, last_frame, scale_percent = 100, fps = 5, filepath='/Users/Arina/', outpath=None):
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
    #fourcc = cv.VideoWriter_fourcc(*'X264')
    video = cv.VideoWriter(outpath, fourcc, fps, (width, height))
    for img in img_array:
        video.write(img)
    video.release()
    cv.destroyAllWindows()



""" Creates and saves a video from raw images for a particular well """
def displayOrigVideo(start_frame, last_frame, filepath, wellNum, outpath='project.avi', fps=5):
    img_array = []
    #for filename in sorted(glob.glob('/Users/Arina/Desktop/02/results/well_1/*.png'), key=numericalSort):
    for i in range(start_frame, last_frame):
        newPath = filepath + "/" + "results/" + "well_" + str(wellNum) +  "/croppedImage_" + str(i + 1) + ".png"
        im = cv.imread(newPath)
        if im is not None:
            img_array.append(im)
        else:
            continue
    frameSize = img_array[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    video = cv.VideoWriter(outpath, fourcc, fps, frameSize, False)   #10 fps
    for img in img_array:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        video.write(img)
    video.release()
    cv.destroyAllWindows()




""" 
Assumptions: 10 fps; 
"""
def plotMAL(major_axis_lengths, MAL = True, title="Plot of MAL over time", outpath = "MAL plot",  show = True):
    if MAL == True:
        time = np.arange(start=0, stop=(len(major_axis_lengths))/5, step = 0.2)
        plt.plot(time, major_axis_lengths)
        plt.title(title)
        plt.ylabel('major axis length, pix')
        plt.xlabel('time, s')
        plt.legend(['MAL, in pix'])
    plt.savefig(outpath)
    plt.show()
    plt.close()
    if not show:
        plt.close('all')


def plotAxes(img):
    """ This is my variation of plotting major/minor axes on the image """
    label_image = skmeasure.label(img)
    axis_major, major_len = data_collection.inertia(label_image, "major")
    axis_minor, minor_len = data_collection.inertia(label_image, "minor")
    x_coord_axis_major = (axis_major[0][0], axis_major[1][0])
    y_coord_axis_major = (axis_major[0][1], axis_major[1][1])
    x_coord_axis_minor = (axis_minor[0][0], axis_minor[1][0])
    y_coord_axis_minor = (axis_minor[0][1], axis_minor[1][1])
    plt.show()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(x_coord_axis_major, y_coord_axis_major,  '-', linewidth=2)
    ax.imshow(img)
    ax.plot(x_coord_axis_minor, y_coord_axis_minor,  '-', linewidth=2)

    """ This is Alex's version """
    axis_major2, inertia, skewness, kurt, vari = data_collection.inertia2(label_image, "major")
    axis_minor2, inertia, skewness, kurt, vari = data_collection.inertia2(label_image, "minor")
    x_coord_axis_major2 = (axis_major2[1][1], axis_major2[0][1])
    y_coord_axis_major2 = (axis_major2[1][0], axis_major2[0][0])
    x_coord_axis_minor2 = (axis_minor2[1][1], axis_minor2[0][1])
    y_coord_axis_minor2 = (axis_minor2[0][0], axis_minor2[1][0])
    plt.show()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(x_coord_axis_major2, y_coord_axis_major2,  '-', linewidth=2)
    ax.imshow(img)
    ax.plot(x_coord_axis_minor2, y_coord_axis_minor2,  '-', linewidth=2)


"""
Creates a pane of images to display
needs some editing 
"""

def showImgs():
    # create figure
    fig = plt.figure(figsize=(50, 35))
    # setting values to rows and column variables
    rows = 6
    columns = 8
    imgs = []
    print(len(imgs))
    # reading images
    for i in range(rows*columns):
        imgs = rin.read_input(48, filepath="/Users/Arina/Desktop/9")
        # Adds a subplot at the i-th position
        fig.add_subplot(rows, columns, i+1)
        # showing image
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title("Well number " + str(i))

#showImgs()



""" 
## Identify the worm
oneWellImg = skan.pre.threshold(cropped)
plt.imshow(np.uint8(cropped))
plt.show()

plt.imshow(np.uint8(oneWellImg))
plt.show()

skeleton = morphology.skeletonize(oneWellImg)
#cv.imshow("skeleton", skeleton)
plt.imshow(skeleton)
plt.show()
#skel_obj = skan.Skeleton(oneWellImg)
#skel_obj.path_lengths(0)


def find_contours():
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  #or cv.CHAIN_APPROX_SIMPLE?
    cnt = contours[0]
    M = cv.moments(cnt) #can be used to calculate other params
    contours = contours[0].reshape(-1, 2)  #Reshape to 2D matrices
    img_copied = img.copy()       #draw the points as individual circles in the image
    for (x, y) in contours:
        cv.circle(img_copied, (x, y), 1, (255, 0, 0), 3)
        cv.imshow("contours", img_copied)
        cv.waitKey(5)



def displaySkeletons(image, skeleton):
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(image, return_distance=True)
    # Compare with other skeletonization algorithms
    skeleton = skeletonize(image)
    skeleton_lee = skeletonize(image, method='lee')
    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skel
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('original')
    ax[0].axis('off')

    ax[1].imshow(dist_on_skel, cmap='magma')
    ax[1].contour(image, [0.5], colors='w')
    ax[1].set_title('medial_axis')
    ax[1].axis('off')

    ax[2].imshow(skeleton, cmap=plt.cm.gray)
    ax[2].set_title('skeletonize')
    ax[2].axis('off')
    ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
    ax[3].set_title("skeletonize (Lee 94)")
    ax[3].axis('off')
    fig.tight_layout()
    plt.show()
"""

""" 

displayFullVideo(1, 1510, scale_percent = 80, fps=10, filepath='/Users/Arina/Downloads/17/17', outpath='/Users/Arina/Downloads/mefl_17_0910_25uM.avi')
displayFullVideo(1, 1510, scale_percent = 80, fps=10, filepath='/Users/Arina/Downloads/18', outpath='/Users/Arina/Downloads/mefl_18_0910_25uM.avi')
displayFullVideo(1, 1510, scale_percent = 80, fps=10, filepath='/Users/Arina/Downloads/19', outpath='/Users/Arina/Downloads/mefl_19_0910_50uM.avi')
"""