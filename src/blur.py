#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge


class Blur:
    def __init__(self):
        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(
            "/lane_det/ll_seg_mask",
            Image,
            self.img_callback,
        )
        self.pub = rospy.Publisher(
            "/lane_det/ll_seg_mask_blur",
            Image,
            queue_size=1,
        )

    def img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        img = cv2.GaussianBlur(img, (7, 7), 0)

        self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="mono8"))


if __name__ == "__main__":
    rospy.init_node("blur")
    flip = Blur()
    rospy.spin()
