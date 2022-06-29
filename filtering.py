
import numpy as np
import skimage.morphology as skmorph
import skimage.measure as skmeasure
from scipy import ndimage
import skimage.segmentation as skseg
import matplotlib.pyplot as plt
import math
from os.path import exists
from os import makedirs
import cv2 as cv
from scipy.stats import skew, kurtosis
from statistics import variance
import filtering


def get_neighbors(arr, row, col):
    neighbors = np.zeros((0, 2))

    if(arr[row-1][col-1]):
        neighbors = np.append(neighbors, [[row-1, col-1]], axis = 0)
    if(arr[row-1][col]):
        neighbors = np.append(neighbors, [[row-1, col]], axis = 0)
    if(arr[row-1][col+1]):
        neighbors = np.append(neighbors, [[row-1, col+1]], axis = 0)

    if(arr[row][col-1]):
        neighbors = np.append(neighbors, [[row, col-1]], axis = 0)
    if(arr[row][col+1]):
        neighbors = np.append(neighbors, [[row, col+1]], axis = 0)

    if(arr[row+1][col-1]):
        neighbors = np.append(neighbors, [[row+1, col-1]], axis = 0)
    if(arr[row+1][col]):
        neighbors = np.append(neighbors, [[row+1, col]], axis = 0)
    if(arr[row+1][col+1]):
        neighbors = np.append(neighbors, [[row+1, col+1]], axis = 0)
    return neighbors



#For a pixel that has 2 neighbors, check if those neighbors are adjacent to each other
def check_adjacent(p1, p2):
    x_dif = abs(p1[0] - p2[0])
    y_dif = abs(p1[1] - p2[1])
    return (x_dif + y_dif <= 1)



def filter_branchpoints(arr):
    new_arr = np.zeros(arr.shape)
    padded = np.pad(arr, pad_width=1, mode="constant", constant_values = 0)

    points = np.argwhere(arr)

    for [i, j] in points:
        neighbors = get_neighbors(padded, i + 1, j + 1)
        if (neighbors.shape[0] > 3):
            new_arr[i][j] = 1
        elif (neighbors.shape[0] == 3):
            if (not (check_adjacent(neighbors[0], neighbors[1]) or check_adjacent(neighbors[0], neighbors[2]) or check_adjacent(neighbors[1], neighbors[2]))):
                new_arr[i][j] = 1
    return new_arr



def filter_endpoints(arr):
    new_arr = np.zeros(arr.shape)
    padded = np.pad(arr, pad_width=1, mode="constant", constant_values = 0)

    points = np.argwhere(arr)

    for [i, j] in points:
        neighbors = get_neighbors(padded, i + 1, j + 1)
        if (neighbors.shape[0] == 1):
            new_arr[i][j] = 1

        if (neighbors.shape[0] == 2):
            if (check_adjacent(neighbors[0], neighbors[1])):
                new_arr[i][j] = 1
    return new_arr




def maxproj(arr):
    return np.amax(arr, axis = 0)



def minproj(arr):
    return np.amin(arr, axis = 0)



#Filter image to only contain largest object
def filter_largest_object(img, leeway, ind=-1):
    labeled = skmeasure.label(img)
    props = skmeasure.regionprops(labeled)
    if (len(props) == 0):
        return img, 0
    maxarea = max([i.area for i in props])
    largest = skmorph.remove_small_objects(labeled.astype(bool), min_size = maxarea-leeway)

    while (sum(sum(largest))==0):  #check if there is at least one object left; a completely empty matrix will sum up to zero
        leeway += 50
        if ind != -1:
            print("adjusting leeway for image at ind", ind, "; new leeway  is ", leeway)
        largest = skmorph.remove_small_objects(labeled.astype(bool), min_size=maxarea - leeway)
    return largest.astype(bool), maxarea




def check_contracted(imgs, index):
    store_images = False

    whole_path = filtering.minproj(imgs)
    proj_diff = filtering.maxproj(imgs) - whole_path
    proj_diff = 255 - proj_diff

    if store_images:
        filteredResultsFolder = "/Users/Arina/Desktop/02/results/testImages/"
        if exists(filteredResultsFolder) is False:
            makedirs(filteredResultsFolder)
            plt.imshow(whole_path)
            plt.savefig(filteredResultsFolder + str(index) + "_01minproj.png")

            plt.imshow(proj_diff)
            plt.savefig(filteredResultsFolder + str(index) + "_02proj_diff.png")

    darkPixels = (proj_diff < 230)

    if store_images:
        plt.imshow(darkPixels)
        plt.savefig(filteredResultsFolder + str(index) + "_03minproj_filtered.png")

    largest, maxarea = filtering.filter_largest_object(darkPixels, 1)
    #print("Contracted check: " + str(maxarea))
    if (maxarea / (imgs.shape[1] * imgs.shape[2]) < 300 / (256 * 320)):
        return True #worm is contracted and swholetill for whole video
    else:
        return False


def generate_disk_filter(imgs_shape):
    disk = skmorph.disk(imgs_shape[1]/2 - 1).astype(int)
    if (disk.shape[0] < imgs_shape[1]):
        disk = np.append(disk, np.zeros((1, disk.shape[1])), axis=0)

    if (disk.shape[1] < imgs_shape[2]):
        diff_left = math.floor((imgs_shape[2] - disk.shape[1])/2)
        diff_right = math.ceil((imgs_shape[2] - disk.shape[1])/2)
        disk = np.concatenate((np.zeros((disk.shape[0], diff_left)), disk, np.zeros((disk.shape[0], diff_right))), axis=1)
    disk = np.invert(disk > 0)

    disks = np.zeros(imgs_shape)
    disks[:] = disk

    return disks


"""
Used on individual thresholded images.
Filters the threshold to be only one object, the one most likely to be a worm.
"""
def get_centermost_big_region(filtered, center_point, index, i, big_enough_ratio, max_area=None):
    store_images = False

    if max_area==None:  # the max size of the object that could be a worm ##TODO: FIX -- not sure why the areas are so  large
        max_area = (filtered.shape[0] * filtered.shape[1]) * 1  # size of the worm cannot be > 0.25 area of the image

    labeled = skmeasure.label(filtered)
    props = skmeasure.regionprops(labeled)

    #check if the area of the object is not the worm
    #print(len(props))
    if (len(props) == 0):
        return filtered

    max_area_found = max([i.area for i in props])
    big_enough_area = max_area_found * big_enough_ratio
    #print([i.area for i in props])
    #print(big_enough_area)

    big_enough_props = []
    for prop_set in props:
        if prop_set.area >= big_enough_area and prop_set.area <= max_area:
            big_enough_props.append(prop_set)
    #print(len(big_enough_props))

    centermost_prop_set = None
    closest_dist = math.inf
    for prop_set in big_enough_props:
        com = prop_set.centroid
        #print(com)
        dist_to_center = np.linalg.norm(np.asarray(center_point) - np.asarray(com))

        if (dist_to_center < closest_dist):
            centermost_prop_set = prop_set
            closest_dist = dist_to_center

    closest_label = centermost_prop_set.label
    centermost = (labeled == closest_label)


    ## CHANGE THIS!!! 
    filteredResultsFolder = "/Users/Arina/Desktop/02/results/testImages/"
    if (store_images and i % 20 == 0):
        plt.imshow(centermost)
        plt.savefig(filteredResultsFolder + str(index) + "_" + str(i) + "_7centermost.png")
    com = ndimage.measurements.center_of_mass(centermost)
    #print(com)
    return np.uint8(centermost), com


def get_centermost(filtered, center_point, big_enough_ratio):
    labeled = skmeasure.label(filtered)
    props = skmeasure.regionprops(labeled)

    if (len(props) == 0):
        return filtered

    max_area = max([i.area for i in props])
    big_enough_area = max_area * big_enough_ratio
    big_enough_props = []
    for prop_set in props:
        if prop_set.area >= big_enough_area:
            big_enough_props.append(prop_set)

    centermost_prop_set = None
    closest_dist = math.inf
    dist_dict = {}
    for prop_set in big_enough_props:
        com = prop_set.centroid
        dist_to_center = np.linalg.norm(np.asarray(center_point) - np.asarray(com))
        curr = prop_set.label
        curr = (labeled == curr)
        dist_dict[dist_to_center] = curr

    return dist_dict





""" 
Used on individual images
if some parts of the worm body are being excluded it can help to get a more complete threshold, but it also has the risk of adding unwanted shadows/dark sections.
"""
def restore_removed_patches(filtered, original, index, i):
    store_images = False

    labeled = skmeasure.label(filtered)
    props = skmeasure.regionprops(labeled)

    if (len(props) == 0):
        return filtered

    bounds = np.asarray(props[0].bbox)

    expand = 25
    bounds[0] = max(bounds[0] - expand, 0)
    bounds[1] = max(bounds[1] - expand, 0)
    bounds[2] = min(bounds[2] + expand, filtered.shape[0])
    bounds[3] = min(bounds[3] + expand, filtered.shape[1])

    bounded = np.zeros(original.shape)
    bounded[:,:] = 255

    bounded[bounds[0]:bounds[2],bounds[1]:bounds[3]] = original[bounds[0]:bounds[2],bounds[1]:bounds[3]]

    if (store_images and i % 20 == 0):
        plt.imshow(bounded)
        plt.savefig("test_images/" + str(index) + "_" + str(i) + "_5bounded.png")

    darkest = np.amin(bounded)
    lightest = np.amax(bounded[bounds[0]:bounds[2],bounds[1]:bounds[3]])

    cutoff = min(darkest + 25, lightest - 25)

    new_filtered = (bounded < cutoff)

    new_filtered = np.logical_or(filtered, new_filtered)

    if (store_images and i % 20 == 0):
        plt.imshow(new_filtered)
        plt.savefig("test_images/" + str(index) + "_" + str(i) + "_6refiltered.png")

    return new_filtered

def filter_images(imgs, no_background, index):
    store_images = False
    #store_images = False
    if store_images:
        filteredResultsFolder = "/Users/Arina/Desktop/02/results/filtered"
        if exists(filteredResultsFolder) is False:
            makedirs(filteredResultsFolder)
        for i in range(0, imgs.shape[0], 20):
            #plt.imshow(imgs[i])
            #plt.savefig(filteredResultsFolder + "/" + "ind" + str(index) + "_i" + str(i)  + "_unfiltered.png")
            impath = filteredResultsFolder + "/" + "ind_" + str(index) + "i_" + str(i) + "_unfiltered.png"
            image = np.uint8(imgs[i])
            cv.imwrite(impath, image)
    disks = generate_disk_filter(imgs.shape)
    worm_contracted = check_contracted(imgs, index)

    if worm_contracted and (no_background == False):   #todo: ask Alex why do we need to check "contracted"?
    #if no_background==False:
        imgs[disks > 0] = 255 #Removes dark corners of images
        darkest = np.amin(imgs, axis=(1,2))   #returns the min value along the (1,2) axis
        cutoffs = darkest + 10 # lower cutoff == stricter params for inclusion
        #cutoffs = darkest + 15
        cutoffs[cutoffs > 100] = 100  # change all vals less than 100 to 100
        filtered = imgs[:] < cutoffs[:, None, None]
    else:
        imgs = 255 - (maxproj(imgs) - imgs)

        if store_images:
            for i in range(0, imgs.shape[0], 20):  #to store only some imgs
            #for i in range(imgs.shape[0]):
                #plt.imshow(imgs[i])
                #plt.savefig(filteredResultsFolder + "/" + "ind_" + str(index) + "i_" + str(i) + "_2subBG.png")
                impath = filteredResultsFolder + "/" + "ind_" + str(index) + "i_" + str(i) + "_2subBG.png"
                image = np.uint8(imgs[i])
                cv.imwrite(impath, image)

        darkest = np.amin(imgs, axis=(1,2))      # darkest (=lowest num) pix value in the image
        cutoffs = darkest + 75
        cutoffs[cutoffs > 230] = 230
        filtered = imgs[:] < cutoffs[:, None, None]   # one image per frame
        """ 
        if sum(sum(imgs))==0:
            print(sum(sum(filtered)), -"-> adjusting")
            darkest = np.amin(imgs, axis=(1, 2))  # darkest (=lowest num) pix value in the image
            cutoffs = darkest + 85
            cutoffs[cutoffs > 230] = 230
            filtered = imgs[:] < cutoffs[:, None, None]  # one image per frame
            print("new: ", sum(sum(filtered)))
        """
    if store_images:
        #for i in range(0, imgs.shape[0], 20):
        for i in range(imgs.shape[0]):
            # plt.imshow(filtered[i])
            # plt.savefig("test_images/" + str(index) + "_" + str(i) + "_3filtered.png")
            # plt.savefig(filteredResultsFolder + "/" + "ind_" + str(index) + "i_" + str(i) +  "_3filtered.png")
            impath = filteredResultsFolder + "/" + "ind_" + str(index) + "i_" + str(i) +  "_3filtered.png"
            image = np.uint8(filtered[i])
            cv.imwrite(impath, image*255)
    return filtered


def refilter(imgs, cutoff_adj = 15):
    disks = generate_disk_filter(imgs.shape)
    imgs[disks > 0] = 255
    darkest = np.amin(imgs, axis=(1, 2))
    cutoffs = darkest + cutoff_adj
    cutoffs[cutoffs > 250] = 250
    filtered = imgs[:] < cutoffs[:, None, None]  # one image per frame
    #nn = np.any(filtered, axis=(1, 2))
    return filtered




#if __name__ == "__main__":
#    main()
