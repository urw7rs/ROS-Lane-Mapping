#!/usr/bin/env python3

import math

import rospy
from sensor_msgs.msg import Image
from lane_det.msg import LanePoints

import numpy as np
from sklearn import cluster

import cv2
from cv_bridge import CvBridge


class Cluster:
    def __init__(self):
        self.bridge = CvBridge()

        self.seg_sub = rospy.Subscriber(
            "/lane_det/ll_seg_mask", Image, self.mask_callback
        )
        self.publisher = rospy.Publisher(
            "/lane_det/ll_points", LanePoints, queue_size=1
        )

        self.dbscan = cluster.DBSCAN(eps=1, min_samples=5, n_jobs=-1)

    def mask_callback(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg)
        mask = mask.astype(np.float32) / 255.0

        self.h, self.w = mask.shape
        self.center = self.w / 2

        bottom = int(self.h * rospy.get_param("/lane_det/top", (480 - 155) / 480))
        middle = int(self.h * rospy.get_param("/lane_det/middle", (480 - 135) / 480))
        top = int(self.h * rospy.get_param("/lane_det/bottom", (480 - 115) / 480))

        self.order = [bottom, middle, top]

        # mask[top:, :] = 0
        # mask[:bottom, :] = 0

        if np.any(mask == 1):
            found_poi = self.cluster_lanes(mask)
            stop_points = self.cluster_stop_line(mask)

            if len(found_poi):
                center_points = self.get_center_lanes(found_poi)

                points = np.concatenate([center_points, stop_points], axis=0)
                points = self.transform_coordinates(points)

                self.publisher.publish(LanePoints(points.flatten().tolist()))

    def cluster_lanes(self, mask):
        y, x = np.where(mask == 1)
        lane_coords = np.stack([y, x]).T

        # label coordinates using DBSCAN
        self.dbscan.fit(lane_coords)

        labels = self.dbscan.labels_

        found_poi = []
        # exclude outlier with label -1
        for label in range(labels.max() + 1):
            # extract coordinates
            (index,) = np.where(labels == label)
            lane_y = y[index]
            lane_x = x[index]

            # average x coordinates
            lane = []
            for unique_y in np.unique(lane_y):
                (y_index,) = np.where(lane_y == unique_y)
                mean_x = int(lane_x[y_index].mean())
                lane.append(np.array([unique_y, mean_x]))
            lane = np.stack(lane)  # [y; x]

            # extract points of interest
            points = np.zeros((3, 2)).astype(int)
            points[:, 0] = self.order
            # points[:, 1].fill(new_unpad_w // 2)
            for i, target_y in enumerate(self.order):
                target_x = np.extract(
                    lane[:, 0] == target_y,
                    lane[:, 1],
                )

                if len(target_x):
                    points[i, 1] = target_x

            found_poi.append(points)

        return found_poi

    def transform_coordinates(self, points):
        tf_poi = points.copy()
        tf_poi[:, :, 1] = tf_poi[:, :, 1] - int(self.center)
        tf_poi[:, :, 0] = self.h - tf_poi[:, :, 0]
        return tf_poi

    def get_center_lanes(self, found_poi):
        found_poi = np.stack(found_poi)

        # distance between center and lane middle point
        distances = np.abs(found_poi[:, 1, 1] - self.center)

        points_of_interest = np.zeros((2, 3, 2)).astype(int)
        points_of_interest[:, :, 0] = self.order
        points_of_interest[:, :, 1].fill(int(self.center))

        # reorder lanes
        index = np.argsort(distances)[:2]
        for i, j in enumerate(index.tolist()):
            points_of_interest[i] = found_poi[j]

        return points_of_interest

    def cluster_stop_line(self, mask):
        points_of_interest = np.zeros((1, 3, 2)).astype(int)
        # n, 2

        mask *= 255
        mask = mask.astype(np.uint8)

        linesP = cv2.HoughLinesP(mask, 1, np.pi / 180, 50, None, 50, 10)

        stoplines = []
        if linesP is not None:
            for i in range(0, len(linesP)):
                x1, y1, x2, y2 = linesP[i][0]

                angle = math.atan2((y2 - y1), (x2 - x1))

                thresh = np.pi / 30
                if (angle > -thresh and angle < thresh) or (
                    angle > np.pi - thresh and angle < np.pi + thresh
                ):
                    stoplines.append(linesP[i][0])

        max_y = 0
        min_y = 1000
        stopline = None
        for line in stoplines:
            xl, yl, xr, yr = line

            ym = (yl + yr) / 2

            if max_y < ym and ym < min_y:
                max_y = ym
                stopline = line

        if stopline is not None:
            xl, yl, xr, yr = stopline
            xm = (xl + xr) / 2
            ym = (yl + yr) / 2

            points_of_interest = np.array([[[xl, yl], [xm, ym], [xr, xl]]], dtype=int)

        return points_of_interest


if __name__ == "__main__":
    rospy.init_node("cluster_lanes", log_level=rospy.DEBUG)
    clustering = Cluster()
    rospy.spin()
