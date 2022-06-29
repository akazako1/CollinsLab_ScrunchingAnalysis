import cv2
import math
import numpy as np
import skimage.morphology as skmorph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatche
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from skimage.morphology.selem import ball
from read_input import read_input
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from PIL import Image
from scipy.stats import skew, kurtosis
from statistics import variance
from time import time
import matplotlib.patches as mpatches

#outputs might be boundary points, skel, COM, head/tail

def calc_midpoint_angles(sorted, midpt, endpoints):
    distances = np.linalg.norm(sorted - midpt, axis = 1)
    distances[distances <= 1.5] = math.inf
    n = 2
    val = np.zeros((n))
    idx = np.zeros((n))

    for i in range(n):
        val[i] = np.amin(distances)
        idx[i] = np.argmin(distances)
        distances[int(idx[i])] = math.inf

    point1 = sorted[int(idx[0])]
    point2 = sorted[int(idx[1])]
    u = [(point1[1] - point2[1]), -(point1[0] - point2[0])]

    num = endpoints.shape[0]
    angles = np.zeros((num))

    for i in range(num):
        v = [(endpoints[i, 0] - midpt[0]), (endpoints[i, 1] - midpt[1])]
        angleCalc = math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])
        angleCalc = abs(angleCalc)

        if angleCalc > math.pi/2:
            angleCalc = math.pi - angleCalc
        angles[i] = angleCalc
    return angles

def calc_endpoint_angles(sorted):

    directions = np.zeros((2))

    if sorted.shape[0] < 5:
        return directions

    spacing = 3

    endpoints = np.asarray([sorted[0], sorted[-1]])
    near_points = np.asarray([sorted[spacing], sorted[-spacing]])

    directions = endpoints - near_points

    angles = np.arctan2(directions[:,0], directions[:,1])

    #print(angles)

    return angles

def point_to_line(midpt, endpoints):
    pt = np.array([midpt[0], midpt[1], 0])
    length = endpoints.shape[0]

    if length == 1:
        return math.nan

    v1 = np.array([endpoints[0,0], endpoints[0,1], 0])
    v2 = np.array([endpoints[1,0], endpoints[1,1], 0])

    a = v1 - v2
    b = pt - v2
    d = np.linalg.norm(np.cross(a, b) / np.linalg.norm(a))

    for i in range(2):
        d_end = math.sqrt(pow(endpoints[i,0] - midpt[0], 2) + pow(endpoints[i,1] - midpt[1], 2))
        if d_end < d:
            d = d_end

    return d

def point_to_line_big(point_list, line):
    pts = np.append(point_list, np.zeros((point_list.shape[0], 1)), axis = 1)
    line = np.append(line, [[0],[0]], axis = 1)

    if line.shape[0] <= 1:
        return np.zeros(point_list.shape[0])

    v1 = line[0]
    v2 = line[1]

    a = v1 - v2
    a = np.reshape(a, (1, 3))
    b = pts - v2

    d = np.linalg.norm(np.cross(a, b) / np.linalg.norm(a), axis = 1)

    return d

def color_stats_body(filtered, image):

    color_points = image[(filtered > 0)]

    #print(color_points.shape)
    #print(color_points)

    if (color_points.shape[0] == 0):
        return 0, 0, 0

    avg_color = np.mean(color_points)
    std_color = np.std(color_points)
    color_diff = np.amax(color_points) - np.amin(color_points)

    #print(avg_color)
    #print(color_diff)

    return avg_color, std_color, color_diff

def color_gradient_skeleton(sorted, image):
    brightness = []

    for i in range(sorted.shape[0]):
        coords = [min(sorted[i][0], image.shape[0] - 1), min(sorted[i][1], image.shape[1] - 1)]
        brightness.append(image[int(coords[0]), int(coords[1])])

    brightness = np.asarray(brightness)

    #print(brightness)

    gradients = np.gradient(brightness)

    #print(gradients)

    total_color_diff = np.amax(brightness) - np.amin(brightness)

    sharpest_gradient = abs(np.amax(np.absolute(gradients)))
    average_gradient = np.mean(np.absolute(gradients))

    return total_color_diff, sharpest_gradient, average_gradient

def smooth_spline(spline, n):
    new_spline = np.zeros(spline.shape)

    new_spline[0:n] = spline[0:n]
    new_spline[new_spline.shape[0]-n:new_spline.shape[0]] = spline[spline.shape[0]-n:spline.shape[0]]
    for i in range(n,spline.shape[0]-n):
        new_spline[i] = (spline[i-n] + spline[i+n]) / 2

    return new_spline

def num_inflection_points(sorted, endpoints):
    sampling_num=5
    if (sorted.shape[0] <= sampling_num):
        return 0

    sample = sorted[::5]
    gradients = np.gradient(sample, axis = 0)
    dydx = np.zeros(gradients.shape[0])
    for i in range(gradients.shape[0]):
        if (gradients[i,0] == 0):
            if (gradients[i,1] > 0):
                dydx[i] = math.inf
            elif (gradients[i,1] < 0):
                dydx[i] = -math.inf
            else:
                dydx[i] = 0
        else:
            dydx[i] = gradients[i,1] / gradients[i,0]

    endpoint_line = endpoints[1] - endpoints[0]
    endpoint_slope = endpoint_line[1] / endpoint_line[0]

    relative_slopes = dydx - endpoint_slope
    acceleration = np.gradient(relative_slopes, axis=0)

    #print(gradients)
    #print(dydx)
    #print(endpoint_slope)
    #print(dydx - endpoint_slope)
    #print(acceleration)

    d_signs = np.sign(acceleration)

    sign_change = 0
    current_sign = 0

    for j in range(d_signs.shape[0]):
        if j == 0:
            current_sign = d_signs[j]
        else:
            if (current_sign != d_signs[j] and d_signs[j] != 0):
                sign_change += 1
                current_sign = d_signs[j]

    return sign_change

def num_crossings(sorted, endpoints, midpt, im_shape):
    gradients = np.gradient(sorted, axis = 0)
    dydx = np.zeros(gradients.shape[0])
    for i in range(gradients.shape[0]):
        if (gradients[i,0] == 0):
            dydx[i] = 0
        else:
            dydx[i] = gradients[i,1] / gradients[i,0]

    d_signs = np.sign(dydx)

    sign_change = 0
    current_sign = 0

    for j in range(d_signs.shape[0]):
        if j == 0:
            current_sign = d_signs[j]
        else:
            if (current_sign != d_signs[j] and d_signs[j] != 0):
                sign_change += 1
                current_sign = d_signs[j]

    num_cross = 0
    divide_factor = 100

    if (sign_change > 1):
        slopeY = (endpoints[0, 1] - endpoints[1, 1]) / divide_factor
        slopeX = (endpoints[0, 0] - endpoints[1, 0]) / divide_factor

        line_points = np.zeros((0, 2))
        for i in range(divide_factor):
            next_pt = [endpoints[1,0] + slopeX * i, endpoints[1,1] + slopeY * i]
            line_points = np.append(line_points, [next_pt], axis = 0)

        change_spline = np.concatenate((sorted, line_points), axis = 0)
        cross_image = np.zeros(im_shape)

        for k in range(change_spline.shape[0]):
            row = min(round(change_spline[k, 0]), cross_image.shape[0] - 1)
            col = min(round(change_spline[k, 1]), cross_image.shape[1] - 1)
            cross_image[int(row), int(col)] = 1
        cross_image = skmorph.skeletonize(cross_image)

        #plt.imshow(cross_image)
        #plt.savefig("test_images/test_image_cross.png")

        branch_num_img = filtering.filter_branchpoints(cross_image)
        crossings = np.argwhere(branch_num_img)

        #plt.imshow(branch_num_img)
        #plt.savefig("test_images/test_image_cross_branches.png")

        #Ignore crossing points near the endpoints
        crossings_elim = np.copy(crossings).astype(float)
        for k in range(crossings_elim.shape[0]):
            dist = math.sqrt(pow(crossings[k,0] - endpoints[0,0], 2) + pow(crossings[k,1] - endpoints[0,1], 2))
            if dist < 15:
                crossings_elim[k,0] = math.nan
            dist = math.sqrt(pow(crossings[k,0] - endpoints[1,0], 2) + pow(crossings[k,1] - endpoints[1,1], 2))
            if dist < 15:
                crossings_elim[k,0] = math.nan

        #If 2 crossing points are right next to each other, only count 1 of them
        for k in range(crossings_elim.shape[0]):
            for l in range(crossings_elim.shape[0]):
                if ((not np.isnan(crossings_elim[k,0])) and (not np.isnan(crossings_elim[l,0]))):
                    dist_cross = math.sqrt(pow(crossings[k,0] - crossings[l, 0], 2) + pow(crossings[k,1] - crossings[l, 1], 2))
                    if (dist_cross > 0 and dist_cross < 10):
                        crossings_elim[l,0] = math.nan

        #print(crossings_elim)

        for k in range(crossings_elim.shape[0]):
            if (not np.isnan(crossings_elim[k,0])):
                num_cross += 1

    return num_cross

def calc_widths(skeleton, image):
    widths = np.zeros((skeleton.shape[0]))

    points = np.zeros((5,2))

    if (skeleton.shape[0] <= 5):
        return widths

    for i in range(skeleton.shape[0]):
        if (i <= 1):
            for j in range(5):
                points[j] = skeleton[j]
            temp_point = points[2]
            points[2] = points[0]
            points[0] = points[2]
        elif (i >= skeleton.shape[0]-2):
            for j in range(5):
                points[j] = skeleton[(skeleton.shape[0]-1) - j]
        else:
            for j in range(5):
                points[j] = skeleton[i + (j - 2)]

        diff = points[-1] - points[0]
        slope = diff[1] / diff[0]

        orthog_slope = -1 / slope

        if orthog_slope > 10:
            orthog_slope = 10
        elif orthog_slope < -10:
            orthog_slope = -10
        elif abs(orthog_slope) < 0.2:
            orthog_slope = 0.1

        x = range(int(points[2,0]-20),int(points[2,0]+20))
        y = orthog_slope * (x - points[2,0]) + points[2,1]

        line_points = np.zeros((y.shape[0], 2))
        line_points[:,0] = x
        line_points[:,1] = y

        left = 10000
        right = 10000
        left_point = np.zeros((1,2))
        right_point = np.zeros((1,2))

        for p in range(line_points.shape[0]):
            direction = points[2] - line_points[p]
            dist = np.linalg.norm(direction)

            row_num = image.shape[0]
            col_num = image.shape[1]
            if (row_num > int(line_points[p,0]) and col_num > int(line_points[p,1]) and int(line_points[0,1]) >= 0 and int(line_points[0,1]) >= 0):
                pixel_val = image[int(line_points[p,0]), int(line_points[p,1])]

                if (direction[0] < 0 and dist < left and pixel_val == 0):
                    left = dist
                    left_point = line_points[p]
                elif (direction[0] > 0 and dist < right and pixel_val == 0):
                    right = dist
                    right_point = line_points[p]

        diff = right_point - left_point
        width = np.linalg.norm(diff)
        widths[i] = width

    #print(widths)

    return widths

def inertia(label_image, axis):
    # Get props
    regions = regionprops(label_image)

    # com = regions.centroid
    if len(regions) == 0:
        print(regions)
        plt.imshow(label_image)
        return 0, 0
    maj_mins = []
    if (axis == "major"):         # Major axis calculation
        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
            # maj_mins = np.array[([[x0, y0], [x2, y2]])]
            maj_mins = [[x0, y0], [x2, y2]]
            axis_len = props.major_axis_length
    else:
        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            # maj_mins = np.array[([[x0, y0], [x1, y1]])]
            maj_mins = [[x0, y0], [x1, y1]]
            axis_len = props.minor_axis_length
    return maj_mins, axis_len



def inertia2(label_image, axis):
    #Get props
    props = regionprops(label_image)

    if len(props) == 0:
        print(props)
        plt.imshow(label_image)
        return 0, 0 , 0, 0, 0
    #Grab major/minor axis values and orientation
    major_axis_rad = props[0].major_axis_length/2
    minor_axis_rad = props[0].minor_axis_length/2
    orientation = math.degrees(props[0].orientation)

    #Get the centroid of the worm
    centroid = props[0].centroid
    x_centroid = centroid[0]
    y_centroid = centroid[1]
    #print(major_axis_rad, minor_axis_rad, centroid, x_centroid, y_centroid)

    if (axis == "major"):
        #Major axis calculation
        maj_x1 = x_centroid + major_axis_rad*math.cos(math.radians(orientation))
        maj_x2 = x_centroid - major_axis_rad*math.cos(math.radians(orientation))
        maj_y1 = y_centroid + major_axis_rad*math.sin(math.radians(orientation))
        maj_y2 = y_centroid - major_axis_rad*math.sin(math.radians(orientation))
        #print(maj_x1)
        ##May want to make this an array of points
        maj_mins = np.array([[maj_x1, maj_y1], [maj_x2, maj_y2]])
    else:
        #Minor axis calculation
        min_x1 = x_centroid + minor_axis_rad*math.cos(math.radians(90 - orientation))
        min_x2 = x_centroid - minor_axis_rad*math.cos(math.radians(90 - orientation))
        min_y1 = y_centroid + minor_axis_rad*math.sin(math.radians(90 - orientation))
        min_y2 = y_centroid - minor_axis_rad*math.sin(math.radians(90 - orientation))
        ##May want to make this an array of points
        maj_mins = np.array([[min_x1, min_y1], [min_x2, min_y2]])

    pointsFilledIM = np.argwhere(label_image)
    allpoints = len(pointsFilledIM)

    #Find distance from every point in thresholded worm to the line between the endpoints
    inertia = 0
    distances = point_to_line_big(pointsFilledIM, maj_mins)

    skewness = skew(distances)
    kurt = kurtosis(distances)
    vari = variance(distances)

    inertia = np.sum(pow(distances, 2))

    #print("skew", skewness, "kurt", kurt, "vari", vari)

    return maj_mins, inertia, skewness, kurt, vari
