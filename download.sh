#!/bin/bash

mkdir models/glpdepth/1
mkdir models/lapdepth/1
mkdir models/hybridnets/1

git clone https://github.com/PINTO0309/PINTO_model_zoo.git

set -e

cd PINTO_model_zoo/245_GLPDepth
bash download_kitti.sh 
cp glpdepth_kitti_480x640/glpdepth_kitti_480x640.onnx ../../models/glpdepth/1/model.onnx

cd ../148_LapDepth
bash download_ldrn_kitti_resnext101.sh
cp saved_model_ldrn_kitti_resnext101_pretrained_data_grad_480x640/ldrn_kitti_resnext101_pretrained_data_grad_480x640.onnx ../../models/lapdepth/1/model.onnx

cd ../276_HybridNets
bash download.sh
cp hybridnets_512x640/hybridnets_512x640.onnx ../../models/hybridnets/1/model.onnx
