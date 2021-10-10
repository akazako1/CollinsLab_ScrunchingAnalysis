
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2 as cv
from os.path import exists
from os import makedirs
import read_input as rin  ## Alex's script
import skan
from skan import pre
from skimage import morphology
import random as rng
import filtering


""" 
# load the files created previously in the MATLAB Mask.m function

#mask_ori = sio.loadmat('/Users/Arina/PycharmProjects/ScrunchingTrack/mask_ori.mat')
#centers = sio.loadmat('/Users/Arina/PycharmProjects/ScrunchingTrack/centers.mat')
centers = np.loadtxt('/Users/Arina/PycharmProjects/ScrunchingTrack/centers.txt')
mask_ori = np.loadtxt('/Users/Arina/PycharmProjects/ScrunchingTrack/mask.txt')


# load all images; create backgrounds
thr = 0.06
plateFolder = "/Users/Arina/Desktop/9"
nFrame = 1500  # 1500 frames for the 5 min video
wells = 48
start=1


I = image.imread(plateFolder + "/" + str(1) + ".jpg")
IasFloat = np.asarray(I,dtype=np.float64)
# I=imread([foldername,'/',num2str(1),'.jpg']);
AVG = np.asarray(I,dtype=np.float64)


for n in range(start, 501):
    filepath = plateFolder + "/" + str(n) + ".jpg"
    I=image.imread(filepath)
    IasFloat = np.asarray(I, dtype=np.float64)
    AVG = AVG+IasFloat
# AVG1=uint8(AVG/500); -- Not necessary?
AVG1=AVG/500

for n in range(501, 1001):
    filepath = plateFolder + "/" + str(n) + ".jpg"
    I=image.imread(filepath)
    IasFloat = np.asarray(I, dtype=np.float64)
    AVG = AVG+IasFloat
AVG2=AVG/500

for n in range(1001, 1501):
    filepath = plateFolder + "/" + str(n) + ".jpg"
    I= image.imread(filepath)
    IasFloat = np.asarray(I, dtype=np.float64)
    AVG = AVG+IasFloat

AVG3=AVG/500    # Create 3 backgrounds (to eliminate the effect of condensation on analysis)

Neg1 = AVG1
Neg2 = AVG2
Neg3 = AVG3

[Height,Width]=AVG1.shape

# Aligh the mask
mask = np.array(mask_ori)

#plt.imshow(np.uint8(AVG1), interpolation='none') # I would add interpolation='none'
#plt.imshow(np.uint8(mask), alpha=0.4, interpolation='none') #  cmap='Reds'/ 'binary'  ##TODO: which colormap to use?

"""

# Crop one well

num_plates = 1
num_wells = 1
frame = 200

# Well cropping params
x_start = 55
y_start = 65
x_end = 1280-2*x_start
y_end = 1024-2*y_start
well_width = 146
well_height = 146

# Files paths/extensions
image_type = ".png"
plateFolder = "/Users/Arina/Desktop/9"
outputPath = '/Users/Arina/PycharmProjects/ScrunchingTrack/'


if exists(plateFolder) is False:
    makedirs(plateFolder)
for j in range(num_wells):
    wellFolder = plateFolder + '/well_' + str(j + 1)
    # wellFolder = './well_data/input/' + output_name + '/well_' + str(i+1)
    if exists(wellFolder) is False:
        makedirs(wellFolder)


imgs = rin.read_input(num_plates, frame, filepath = "/Users/Arina/Desktop/9")

plate = 1
for i in range(num_plates):
    plate += 1
    #for frame, img in enumerate(imgs[plate - 1]):
    for frame, img in enumerate(imgs):
        if img is not None:
            counter = 0
            # this should be reversed so counter%48 can give the well num
            for i in range(x_start, x_end, well_width):
                for j in range(y_start, y_end, well_width):
                    counter += 1
                    x_1 = x_start + (i - x_start)
                    y_1 = y_start + (j - y_start)
                    y_2 = y_1 + well_height
                    x_2 = x_1 + well_height
                    cropped = img[y_1:y_2, x_1:x_2]
                    impath = plateFolder + "/" + "well_" + str(counter) + "/" + "croppedImage" + "_" + str(frame + 1) + image_type
                    cv.imwrite(impath, cropped)
                    #well_imgs.insert(counter, cropped)  # todo: check what do we store
                    #plt.imsave(impath, cropped)



well_imgs  = rin.read_input_oneWell(frame, filepath= "/Users/Arina/Desktop/9", wellNum = 1)

##TODO


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
"""



if (well_imgs.shape[-1] == 3):  ## reshape
    well_imgs_reshaped = well_imgs[:, :, :, 0]

filtered_imgs = filtering.filter_images(well_imgs_reshaped, False, 1)

image_dims = (imgs.shape[1], imgs.shape[2]) ## the size of one well

""" 

# TODO: Remove background and identify the worm
# Loop through each frame of the current video
for i in range(0, imgs.shape[0], 20):  #for every frame    use e.g, imgs[1,:,:] to get one frame
    index = int(i/20)
    #for i in range(len(filtered_imgs)):
    #plt.imshow(filtered_imgs[i])
    impath = plateFolder + "/" + "well_" + str(index+1) + "/" + "filtered"+ image_type
    #image = np.array(filtered_imgs[index])
    #cv.imwrite(impath, image)
    plt.imshow(image)
    plt.savefig(impath)


"""