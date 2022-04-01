#!/usr/bin/env python3

import sys


import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class Process:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )

        self.ll_seg_pub = rospy.Publisher("/lane_det/ll_seg_mask", Image, queue_size=1)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        msg = self.bridge.cv2_to_imgmsg(ll_seg_mask.astype(np.uint8))


if __name__ == "__main__":
    rospy.init_node("model")
    model = Model()
    rospy.spin()
