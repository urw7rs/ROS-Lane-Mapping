FROM osrf/ros:noetic-desktop-full

WORKDIR /catkin_ws/src
WORKDIR /catkin_ws

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make

RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache \
    opencv-python \
    tritonclient[all] \
    scikit-learn
