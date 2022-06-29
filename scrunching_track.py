
"""
This is the main scrunching tracking script
It assumes that the raw data has been preprocessed: individuals wells cropped
and the images of individual wells were saved in the corresponding folders
"""
import numpy as np
from scipy import ndimage
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal
import cv2 as cv
import read_input as rin  ## Alex's script
from skimage.measure import label, regionprops
import filtering
import data_collection
from statistics import stdev, mean



def body_len(mal_arr, interval):
    body_len_arr = []
    for i in range(0, len(mal_arr), interval):
        body_len_arr.append(np.nanmean(mal_arr[i:i+interval]))
    return body_len_arr

def calculate_velocities(com_arr, mal_arr, fps=5):
    velocities = []
    mean_displacements = []
    all_displacements = []
    blen_arr = body_len(mal_arr, interval=30*fps)
    for j in range(int(len(com_arr)/(30*fps))):
        disp_arr = []
        for i in range(j*30*fps, (j+1)*30*fps): # 30 s intervals
            #print(i + 6*fps)
            curr_disp = np.linalg.norm(com_arr[i + 6*fps] - com_arr[i])  # com_arr[i + 60] is the frame 6 second after the current frame
            if ~np.isnan(curr_disp):
                disp_arr.append(curr_disp)
        disp_arr = np.array(disp_arr)
        velocity = np.nansum(disp_arr)/np.count_nonzero(~np.isnan(disp_arr))
        velocities.append(velocity)
        all_displacements.append(disp_arr)
        mean_displacements.append(mean(disp_arr)/blen_arr[0])
    return velocities, mean_displacements, all_displacements


def select_closest(img, com_arr, center_point, last_non_nan_ind, fr=0.3, max_displacement=100):
    dd = filtering.get_centermost(img, center_point, fr)
    #print(dd.keys())
    largest_indx = sorted(dd)[:7]
    min_dist = 10000
    min_dist_ind = 0
    for ind in largest_indx:
        # pick the one closest
        com = ndimage.measurements.center_of_mass(dd[ind])
        disp = np.linalg.norm(
            np.array(com_arr[last_non_nan_ind]) - np.array(com))  # Euclidean  distance; need to conver to np arrays from tuples
        if min_dist > disp:
            min_dist_ind = ind
            min_dist = disp
    if min_dist < max_displacement:
        centermost = dd[min_dist_ind]
        return np.uint8(centermost)
    else:
        return False


def calculate_worm_size(areas_arr):
    cleaned = []
    # remove clear outliers
    for i in range(len(areas_arr)):
        if (~np.isnan(areas_arr[i])) and (areas_arr[i] > 10) and (areas_arr[i] < 1000):
            #print("tracked area ", tracked_areas[i])
            cleaned.append(areas_arr[i])
        else:
            #print("area too big/small:", tracked_areas[i])
            continue
    if len(cleaned) == 0 or (sum(cleaned) / len(cleaned))<250:
        if not len(cleaned) == 0:
            print("actual worm size ==", sum(cleaned) / len(cleaned))
        return 500 #todo-- this is a termporary solution
    return sum(cleaned) / len(cleaned)

# mean of worm aspect ratio (length^2/area) during the oscillation  > 6  (usually 8~13 for normally glidign worm).
def calculate_asp_ratios(centermost_arr):  #todo; make sure indexes match
    asp_ratio_arr = []
    for centermost in centermost_arr:
        if sum(sum(centermost)) != 0:
            contours, _ = cv.findContours(centermost, 1, 2)
            cnt = contours[0]
            _, (width, height), _ = cv.minAreaRect(cnt)
            """
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect) 
            box = np.int0(box)
            img = cv.drawContours(centermost, [box], -1, (255, 0, 0), -1)
            plt.imshow(img)
            """
            if width>height:
                asp_ratio_arr.append(width/height)
            else:
                asp_ratio_arr.append(height/width)
        else:
            asp_ratio_arr.append(np.nan)
    return asp_ratio_arr


"""
Analyzes video for one well 


Returns arrays with frame-by-frame information for the well:
    centermost_arr - an array containing the XXXXXX
    mal_arr - major axis length 
    com_arr - Center of Mass tracking 
    asp_ratio_arr - Aspect Ratio value

"""
def analyze(wellNum, plateFolder, start_frame, end_frame):
    index = 1  # todo: change
    end_frame -= 1 #todo fix
    big_enough_ratio = 0.3
    max_displacement = 120  # max displacement between two frames (in pix)  # todo: chenge?

    outputPath = plateFolder + "/results"
    print("Starting to read images for well "+str(wellNum))
    well_imgs = rin.read_input_oneWell(start_frame, end_frame, filepath=plateFolder, wellNum=wellNum)

    if well_imgs.shape[0]>end_frame:
        print("check the that all imgs exist")
    if well_imgs.shape[-1] == 3:
        well_imgs = well_imgs[:, :, :, 0]

    filtered_imgs = filtering.filter_images(well_imgs, no_background=False, index=1)

    nn = np.any(filtered_imgs, axis=(1, 2))  # False means that that image doesn't have an object in it
    lost_indx = np.where(~nn)
    reanalyze = well_imgs[lost_indx]

    refiltered_imgs = filtering.refilter(reanalyze, cutoff_adj=20)
    filtered_imgs[tuple(lost_indx)] = refiltered_imgs

    image_dims = (filtered_imgs[0].shape[0], filtered_imgs[0].shape[1])  # size of one well
    center_point = [filtered_imgs[0].shape[0] / 2, filtered_imgs[0].shape[1] / 2]

    #tracked_areas = []
    mal_coord_arr = []
    mal_arr = []
    centermost_arr = []
    com_arr = []
    asp_ratio_arr = []

    # initial read
    img = np.array(filtered_imgs[0])
    centermost, com = filtering.get_centermost_big_region(filtered_imgs[0], center_point, index, 0, big_enough_ratio, max_area=None)
    com_arr.append(com)
    if np.any(centermost):
        _, mal = data_collection.inertia(label(centermost), "major")
        if mal<90:
            mal_arr.append(mal)
        else:
            mal_arr.append(60)
        # tracked_areas.append(sum(sum(centermost)))
    else:
        mal_arr.append(np.nan)
    centermost_arr.append(centermost)

    last_ind = 0
    curr_discarded = 0
    total_discarded = 0
    discarded_hist = {}  # key: first discarded frame; value: number of discarded frames
    empty_frame = np.zeros((image_dims), dtype=bool)

    print("Finished reading the images")

    # Calculate the average size of the worm based on the first 100 frames
    temp_areas_arr = []
    for i in range(0, 100):  # for every frame    use e.g, imgs[1,:,:] to get one frame
        centermost, com = filtering.get_centermost_big_region(filtered_imgs[i], center_point, index, i, big_enough_ratio, max_area=None)
        disp = np.linalg.norm(
            com_arr[last_ind] - np.array(com))  # Euclidean  distance; need to conver to np arrays from tuples
        if disp > 50:
            #print("too far away with displacement", disp)
            centermost = select_closest(img, com_arr, center_point, last_non_nan_ind=last_ind, fr=big_enough_ratio,
                                        max_displacement=max_displacement)
        temp_areas_arr.append(sum(sum(centermost)))

    av_worm_size = calculate_worm_size(temp_areas_arr)
    #print("average worm size: ", av_worm_size)

    for i in range(1, filtered_imgs.shape[0]):  # for every frame    use e.g, imgs[1,:,:] to get one frame
        img = np.array(filtered_imgs[i])
        if sum(sum(img)) < 0:  # check if there is at least one object left
            print("no object")

        centermost, com = filtering.get_centermost_big_region(filtered_imgs[i], center_point, index, i, big_enough_ratio, max_area=None)

        # check if the COM of the object is too far away for it to be a worm:
        disp = np.linalg.norm(com_arr[last_ind] - np.array(com))  # Euclidean  distance; need to conver to np arrays from tuples

        if disp > 50 and curr_discarded < 5:
            centermost = select_closest(img, com_arr, center_point, last_non_nan_ind=last_ind, fr=big_enough_ratio,
                                        max_displacement=max_displacement)

        if (np.any(centermost)) and (sum(sum(np.array(centermost))) < av_worm_size * 2):
            # largest, maxarea = filtering.filter_largest_object(img, leeway=100, ind=i)
            # label_image = label(largest)
            label_image = label(centermost)
            # axis_major, inertia, skewness, kurt, vari = data_collection.inertia2(label_image, "major")
            #axis_minor, inertia, skewness, kurt, vari = data_collection.inertia2(label_image, "minor")
            mal_coord, mal = data_collection.inertia(label_image, "major")
            _, minor = data_collection.inertia(label_image, "minor")
            # largest_arr.append(np.uint8(largest))
            centermost_arr.append(np.uint8(centermost))
            # tracked_areas.append(maxarea)
            mal_coord_arr.append(mal_coord)
            com_arr.append(com)
            mal_arr.append(mal)
            last_ind = len(mal_arr)-1  # store the last ind of the non-nan elem

            if curr_discarded > 0:
                discarded_hist[i - curr_discarded] = curr_discarded
                curr_discarded = 0

            if minor > 0:
                asp_ratio_arr.append(mal/minor)
            else:
                asp_ratio_arr.append(np.nan)
        else:
            com_arr.append((np.nan, np.nan))
            # largest_arr.append(np.nan)
            centermost_arr.append(empty_frame)  # uncomment unless making a movie
            # tracked_areas.append(np.nan)
            mal_coord_arr.append(np.nan)
            mal_arr.append(np.nan)
            asp_ratio_arr.append(np.nan)

            curr_discarded += 1
            total_discarded += 1

    print(total_discarded, "(", int(total_discarded / (end_frame - start_frame) * 100), "%) of frames for well", wellNum, "were discarded")
    return centermost_arr, mal_arr, com_arr, asp_ratio_arr


"""

lgnd = []
for i, com_arr in enumerate(COMs):
    leg.append("Well " + str(wells[i]))
    velocities = calculate_velocities(com_arr)
    time = np.arange(start=15, stop=(len(com_arr)/10-15), step=30)
    plt.scatter(time, velocities)
    plt.plot(time, velocities, linestyle='--')
    plt.xlabel('time, seconds')
    plt.ylabel('Mean velocity, pixels')
    plt.xlim((0, (len(com_arr)/10)-15))
    #plt.title("Well " + str(wells[i]))
    #outpath = os.path.expanduser("/Users/Arina/Desktop/02/results/peak_sets/peak sets well" + str(wells[i]) + ".png")
    #plt.savefig(outpath)
    plt.show()
plt.legend(lgnd)
plt.close(0)
"""
