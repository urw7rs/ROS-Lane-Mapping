#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


class Process:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/lane_mapping/ll_seg_mask", Image, self.image_callback
        )

        self.publisher = rospy.Publisher("/lane_mapping/centers", Image, queue_size=1)

        self.threshold = 250

    def get_center(self, mask, y):
        h, w = mask.shape

        (x,) = np.where(mask[y] > self.threshold)

        x_left = x_right = 0.0

        if len(x) > 0:
            left = x[x < w // 2]
            if len(left) > 0:
                x_left = np.max(x[x < w // 2])

            right = x[x > w // 2]
            if len(right) > 0:
                x_right = np.min(right)

        return x_left, x_right

    def get_diff(self, pts):
        degs = []
        dists = []
        for i in range(2):
            dx, dy = pts[i] - pts[i + 1]

            rad = np.arctan2(dy, dx)
            deg = rad / np.math.pi * 180

            dist = np.sqrt(dx ** 2 + dy ** 2)

            degs.append(deg)
            dists.append(dist)

        deg = np.array(degs)
        dist = np.array(dists)

        return deg, dist

    def image_callback(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        h, w = mask.shape

        left_pts = []
        right_pts = []
        for r in [0.57, 0.61, 0.65]:
            y = int(h * r)
            x_left, x_right = self.get_center(mask, y)

            left_pts.append((x_left, y))
            right_pts.append((x_right, y))

            breakpoint()
            if x_left is not None and x_right is not None:
                cv2.circle(mask, (int(x_left), y), 10, 200, 3)
                cv2.circle(mask, (int(x_right), y), 10, 200, 3)
                cv2.circle(mask, (int((x_left + x_right) / 2), y), 10, 100, 3)

        left_pts = np.array(left_pts)
        right_pts = np.array(right_pts)

        left_deg, left_dist = self.get_diff(left_pts)
        right_deg, right_dist = self.get_diff(right_pts)

        for i in range(2):
            cv2.putText(
                mask,
                f"{left_deg[i]} {left_dist}",
                left_pts[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                mask,
                f"{right_deg[i]} {right_dist}",
                right_pts[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8))

        self.publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node("model")
    process = Process()
    rospy.spin()
