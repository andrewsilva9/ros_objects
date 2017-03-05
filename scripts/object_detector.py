import cv2
import numpy as np


def find_objects(input_image):
    """
    :param input_image: image imported for use by CV2
    :return: a list of bounding box arrays of the form [[x1, y1, width, height], [x1, y1, width, height], ...]
    """
    # Only want bottom half of the image
    down_shift = len(input_image) / 2
    img = input_image[down_shift:, :, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold and only allow super white through (table)
    ret, th1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    final_cons = []
    for con in bounding_boxes:
        area = con[2] * con[3]
        # If box is too large
        if area > 3000:
            continue
        # If box is too small
        if area < 100:
            continue
        mostly_white = area * 240
        # If box is too white
        if np.sum(gray[con[1]:con[1] + con[3], con[0]:con[0] + con[2]]) > mostly_white:
            continue
        # TODO: if this is a tablet, continue...
        # TODO: get better at ignoring cozmo boxes?
        appender = [con[0], con[1] + down_shift, con[2], con[3]]
        final_cons.append(appender)
    return final_cons
