#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge


class Flip:
    def __init__(self):
        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.img_callback,
        )
        self.pub = rospy.Publisher(
            "/lane_mapping/image",
            Image,
            queue_size=1,
        )

    def img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)

        self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))


if __name__ == "__main__":
    rospy.init_node("flip")
    flip = Flip()
    rospy.spin()
