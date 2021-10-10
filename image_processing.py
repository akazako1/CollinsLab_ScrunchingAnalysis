import numpy as np
import skimage.morphology as skmorph
import skimage.measure as skmeasure
from skimage.measure import regionprops, label
import math
from scipy.stats import skew, kurtosis
from statistics import variance


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
    #print([i.area for i in props])
    #print(maxarea)
    largest = skmorph.remove_small_objects(labeled.astype(bool), min_size = maxarea-leeway)

    while (sum(sum(largest))==0):  #check if there is at least one object left; a completely empty matrix will sum up to zero
        leeway += 50
        if ind != -1:
            print("adjusting leeway for image at ind", ind, "; new leeway  is ", leeway)
        largest = skmorph.remove_small_objects(labeled.astype(bool), min_size=maxarea - leeway)
    #print(np.count_nonzero(largest))
    return largest.astype(bool), maxarea
