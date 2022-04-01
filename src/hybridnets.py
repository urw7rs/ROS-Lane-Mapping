#!/usr/bin/env python3

import sys


import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class ParseTritonModel:
    def __init__(self, model_name: str, client):
        """
        input [
          {
            name: "images"
            data_type: TYPE_FP32
            format: FORMAT_NCHW
            dims: [ 3, 640, 640]
            reshape { shape: [ 1, 3, 640, 640] }
          }
        ]
        output [
          {
            name: "det_out"
            data_type: TYPE_FP32
            dims: [ 1, 6300, 6 ]
          },
          {
            name: "drive_area_seg"
            data_type: TYPE_FP32
            dims: [ 1, 2, 640, 640 ]
          },
          {
            name: "lane_line_seg"
            data_type: TYPE_FP32
            dims: [ 1, 2, 640, 640 ]
          }
        ]
        """

        self.model_name = model_name
        self.client = client

        try:
            model_metadata = client.get_model_metadata(model_name, as_json=True)
        except InferenceServerException as e:
            print(f"failed to get {model_name} model metadata: {str(e)}")
            sys.exit(1)

        self.input_metadata = model_metadata["inputs"]
        self.output_metadata = model_metadata["outputs"]

    def generate_input(self, *inputs):
        return self._generate_input(*inputs)

    def _generate_input(self, *inputs):
        infer_inputs = []
        for metadata, np_input in zip(self.input_metadata, inputs):
            infer_input = grpcclient.InferInput(
                metadata["name"],
                np_input.shape,
                metadata["datatype"],
            )
            infer_input.set_data_from_numpy(np_input)

            infer_inputs.append(infer_input)
        return infer_inputs

    def generate_req_output(self):
        return self._generate_req_output()

    def _generate_req_output(self):
        req_output = []
        for metadata in self.output_metadata:
            req_output.append(grpcclient.InferRequestedOutput(metadata["name"]))
        return req_output

    def parse_infer_result(self, infer_result):
        return self._parse_infer_result(infer_result)

    def _parse_infer_result(self, infer_result):
        result = []
        for metadata in self.output_metadata:
            parsed_result = infer_result.as_numpy(metadata["name"])
            if metadata["datatype"] == "BYTES":
                parsed_result = list(
                    map(lambda t: t.decode("utf-8"), parsed_result.tolist())
                )
                if len(parsed_result) == 1:
                    parsed_result = parsed_result[0]
            result.append(parsed_result)

        if len(result) == 1:
            return result[0]
        else:
            return result


class SyncTritonModel(ParseTritonModel):
    def __init__(self, model_name, client):
        super().__init__(model_name, client)

    def __call__(self, *inputs):
        infer_result = self.client.infer(
            model_name=self.model_name,
            inputs=self.generate_input(*inputs),
            outputs=self.generate_req_output(),
        )
        return self.parse_infer_result(infer_result)


class AsyncTritonModel(ParseTritonModel):
    def __init__(self, model_name, client):
        super().__init__(model_name, client)

    def __call__(self, *inputs):
        inputs = self.generate_input(*inputs)
        outputs = self.generate_req_output()
        self.client.async_infer()


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)

    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh : dh + new_unpad_h, dw : dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


class Model:
    def __init__(self, client, model_name="hybridnets"):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/lane_det/image", Image, self.image_callback)
        self.ll_seg_pub = rospy.Publisher("/lane_det/ll_seg_mask", Image, queue_size=1)

        self.model = SyncTritonModel(model_name, client)
        metadata = self.model.input_metadata[0]["shape"]

        self.img_shape = [int(s) for s in metadata[1:]]

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        height, width, _ = img.shape

        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(
            img, self.img_shape
        )

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        regression, classification, segmentation = self.model(img)

        # select ll segment area.
        ll_seg_mask = segmentation[:, :, dh : dh + new_unpad_h, dw : dw + new_unpad_w]
        ll_seg_mask = np.argmax(ll_seg_mask, axis=1)  # (?,?) (0|1)
        ll_seg_mask[ll_seg_mask == 1] = 0

        ll_seg_mask *= 255
        ll_seg_mask = ll_seg_mask.transpose(1, 2, 0)

        msg = self.bridge.cv2_to_imgmsg(ll_seg_mask.astype(np.uint8), encoding="mono8")
        self.ll_seg_pub.publish(msg)


if __name__ == "__main__":
    try:
        client = grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
        )
    except InferenceServerException as e:
        print(f"failed to creat context {str(e)}")
        sys.exit(1)

    rospy.init_node("hybridnets")
    model = Model(client)
    rospy.spin()
