# !/usr/bin/env python

"""
Object Detector Object for ROS
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
os.environ['GLOG_minloglevel'] = '2'

import caffe


class ObjectDetector:
    def __init__(self):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        prototxt = os.path.join(cfg.MODELS_DIR, 'coco_blocks',
                                'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'coco_blocks_faster_rcnn_final.caffemodel')

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))
        # TODO set this to be my list of classes
        self.class_list = ['__background__',
                   'person', 'backpack', 'bottle', 'cup',
                   'bowl', 'banana', 'apple', 'orange', 'pizza', 'donut',
                   'tv', 'laptop', 'cell phone', 'book', 'screw', 'block', 'beam']

        self.my_model_dir = '/home/andrewsilva/faster_rcnn/py-faster-rcnn/models'
        self.my_output_dir = '/home/andrewsilva/faster_rcnn/py-faster-rcnn/output/default/train'
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        print '\n\nLoaded network {:s}'.format(caffemodel)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(1):
            _, _ = im_detect(self.net, im)

    def find_objects(self, input_image):
        # input_image = input_image.astype(float)
        scores, boxes = im_detect(self.net, input_image)
        # Visualize detections for each class
        objects_detected = []
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(self.class_list[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            dets = dets[dets[:, -1] >= CONF_THRESH]
            dets[:, -1] = cls_ind
            # append data structure with format [[x1, y1, x2, y2, obj_id], [x1, y1, x2, y2, obj_id], ...] for all boxes
            objects_detected.append(dets)
        return objects_detected

