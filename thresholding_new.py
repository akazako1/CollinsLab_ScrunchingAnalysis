
import cv2
import numpy as np


def get_threshold_frame(frame, threshold, remove_noise=False, min_threshold=None, dilate=False):
    _, threshold_frame = cv2.threshold(frame.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)
    if min_threshold is not None:
        _, threshold_frame_2 = cv2.threshold(frame.astype(np.uint8), min_threshold, 255, cv2.THRESH_BINARY_INV)
        threshold_frame = np.logical_and(threshold_frame, np.logical_not(threshold_frame_2)).astype(np.uint8)*255
    np.divide(threshold_frame, 255, out=threshold_frame, casting='unsafe')

    # optionally remove noise from the thresholded image
    if remove_noise:
        kernel = np.ones((3, 3), np.uint8)
        threshold_frame = cv2.erode(threshold_frame, kernel, iterations=1)
        threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=1)

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        threshold_frame = cv2.erode(threshold_frame, kernel, iterations=1)
        threshold_frame = cv2.dilate(threshold_frame, kernel, iterations=2)

    return threshold_frame



def extract_body_position_extended(frame, eyes_threshold_value, body_threshold_value, threshold_value = None, threshold_step = 1, erode = False, kernel_size = [3, 3], n_iterations = 1):
    if threshold_value == None:
        threshold_value = eyes_threshold_value
    body_position = None
    try:
        if threshold_value < body_threshold_value:
            threshold_frame = get_threshold_frame(frame, threshold_value)
            if erode:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1]))
                threshold_frame = cv2.erode(threshold_frame, kernel, iterations = n_iterations)
            contours = cv2.findContours(threshold_frame.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            if len(contours) == 3:
                moments = [cv2.moments(contours[i]) for i in range(len(contours))]
                if np.min(np.array([moments[i]['m00'] for i in range(len(moments))])) < 1:
                    threshold_value += threshold_step
                    body_position = extract_body_position_extended(frame = frame, eyes_threshold_value = eyes_threshold_value, body_threshold_value = body_threshold_value, threshold_value = threshold_value, threshold_step = threshold_step)
                elif body_position is None:
                    body_position = np.array([np.average([moments[i]['m01'] / moments[i]['m00'] for i in range(len(moments))]), np.average([moments[i]['m10'] / moments[i]['m00'] for i in range(len(moments))])])
            else:
                threshold_value += threshold_step
                body_position = extract_body_position_extended(frame = frame, eyes_threshold_value = eyes_threshold_value, body_threshold_value = body_threshold_value, threshold_value = threshold_value, threshold_step = threshold_step)
        else:
            body_position = None
    except:
        body_position = None
    return body_position
