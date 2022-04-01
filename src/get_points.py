#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


num_horizon_points = 0
new_horizon_points = []


def get_horizon_point(event, x, y, flags, param):
    global num_horizon_points, new_horizon_points

    if event == cv2.EVENT_LBUTTONDBLCLK:
        new_horizon_points.append([x, y])
        num_horizon_points += 1


class GetPoints:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/lane_det/ll_seg_mask", Image, self.image_callback
        )

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Draw horizontal line
        img = cv2.line(
            img,
            (0, img.shape[0] // 2),
            (img.shape[1], img.shape[0] // 2),
            (0, 0, 251),
            1,
        )

        cv2.imshow("Get horizon points", img)

        if cv2.waitKey(1) == ord("q"):
            global num_horizon_points, new_horizon_points

            num_points = 0
            while True:
                if num_points < num_horizon_points:
                    x, y = new_horizon_points[num_horizon_points - 1]
                    img = cv2.circle(
                        img,
                        (x, y),
                        5,
                        (251, 191, 67),
                        -1,
                    )

                    cv2.imshow("Get horizon points", img)
                    num_points += 1
                elif num_points == 4:
                    horizon_points = np.float32(new_horizon_points)

                    num_horizon_points = 0
                    new_horizon_points = []
                    break

                cv2.waitKey(100)

            print(f"horizon_points = np.{repr(horizon_points)}")


if __name__ == "__main__":
    cv2.namedWindow("Get horizon points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Get horizon points", get_horizon_point)

    rospy.init_node("get_points")
    model = GetPoints()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyWindow("Get horizon points")
