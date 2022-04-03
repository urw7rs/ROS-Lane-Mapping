# ROS nodes for creating Lane Detection and Mapping

## Quick Start

### prepare model repository

download onnx files from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) by running `./download.sh`

copy `glpdepth_kitti_480x640` to `models/glpdepth/1/model.onnx`
copy `saved_model_ldrn_kitti_resnext101_pretrained_data_grad_480x640` to `models/lapdepth/1/model.onnx`
copy `hybridnets_512x640/hybridnets_512x640.onnx` to `models/hybridnets/1/model.onnx`


### start triton inference server

run `$ docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:22.03-py3 tritonserver --model-repository=/models`

### run ros nodes

**Run on Docker if you can't install ROS noetic**

1. build container

`docker build -t ros .`

2. run container with device and display access
```
xhost +
docker run --rm \
    --privileged \
    --net host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/catkin_ws/src/lane_mapping \
    -it ros bash
```

**launch ros nodes**

run `catkin_make` in workspace root directory to build msgs

1. select 4 of corners of the lane to calibrate the camera

run `roslaunch lane_mapping calib.launch` to launch calibration nodes an opencv window will open press `q` to select 4 points

select in the order of lower left, lower right, upper left, upper right. the transformation matrix will be printed to stdout. copy the matrix and paste it to the `src_pts` numpy array in src/ipm.py 

2. launch the main ros nodes

run `roslaunch lane_mapping detect.launch` or `roslaunch lane_mapping detect.launch` if camera is flipped
