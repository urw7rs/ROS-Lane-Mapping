#!/usr/bin/env python3

import rospy
from lane_mapping.msg import LanePoints

import socket


class Broadcaster:
    def __init__(self):
        self.seg_sub = rospy.Subscriber(
            "/lane_mapping/ll_points", LanePoints, self.point_callback
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def point_callback(self, msg):
        tf_poi = msg.data

        HEADER = "$CAM_LANE"
        message = [HEADER]
        for i in range(0, len(tf_poi) // 2, 2):
            message.append(f"{tf_poi[i]} {tf_poi[i]}")
        message = ", ".join(message) + ","

        ip = rospy.get_param("/lane_mapping/ip", "127.0.0.1")
        port = rospy.get_param("/lane_mapping/port", 5005)
        self.sock.sendto(message.encode(), (ip, port))


if __name__ == "__main__":
    rospy.init_node("broadcast")
    broadcaster = Broadcaster()
    rospy.spin()
