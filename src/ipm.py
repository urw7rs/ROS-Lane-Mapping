#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


src_pts = np.array(
    [
        [25.0, 473.0],
        [500.0, 478.0],
        [272.0, 260.0],
        [308.0, 259.0],
    ],
    dtype=np.float32,
)


class IPM:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/lane_mapping/ll_seg_mask_blur", Image, self.image_callback
        )
        self.ipm_pub = rospy.Publisher(
            "/lane_mapping/ipm",
            Image,
            queue_size=1,
        )

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        w, h = img.shape

        scale = 1
        offset = w * 0.4
        dst_pts = np.float32(
            [
                [offset, h * scale],
                [w - offset, h * scale],
                [offset, 0],
                [w - offset, 0],
            ]
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev = cv2.warpPerspective(img, M, (w, int(h * scale)))

        msg = self.bridge.cv2_to_imgmsg(bev, encoding="mono8")
        self.ipm_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("IPM")
    model = IPM()
    rospy.spin()
