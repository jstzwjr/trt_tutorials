#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 9:16
# @Author  : wangjianrong
# @File    : batchedNMS_test.py


import sys
sys.path.insert(0,"/workspace/projects/trt_tutorials")

from base.myutils import build_engine,load_trt_engine,TRT_LOGGER
import tensorrt as trt
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from base import common
import os


trt.init_libnvinfer_plugins(TRT_LOGGER,'')



# 创建并报错只有batchnms一个节点的onnx模型，并转tensorrt模型

num_classes = 10
keepTopK = 100
shareLocation = True
batch_size = 1
num_boxes = 2000

node = gs.Node(
    op="BatchedNMS_TRT",
    attrs={
        "shareLocation": shareLocation,
        "backgroundLabelId": -1,
        "numClasses": num_classes,
        "topK": 1024,
        "keepTopK": keepTopK,
        "scoreThreshold": 0.0,
        "iouThreshold": 0.7,
        'isNormalized': True,
        'clipBoxes': True,
        "scoreBits":10},
)

if shareLocation:
    boxes = gs.Variable("boxes",dtype=np.float32,shape=(batch_size,num_boxes,1,4))
else:
    boxes = gs.Variable("boxes",dtype=np.float32,shape=(batch_size,num_boxes,num_classes,4))
scores = gs.Variable("scores",dtype=np.float32,shape=(batch_size,num_boxes,num_classes))

nms_num_detections = gs.Variable(name="nms_num_detections", dtype=np.int32, shape=(batch_size, 1))
nms_boxes = gs.Variable(name="nms_boxes", dtype=np.float32, shape=(batch_size, keepTopK, 4))
nms_scores = gs.Variable(name="nms_scores", dtype=np.float32, shape=(batch_size, keepTopK))
nms_classes = gs.Variable(name="nms_classes", dtype=np.float32, shape=(batch_size, keepTopK))

node.inputs = [boxes, scores]
node.outputs = [nms_num_detections, nms_boxes, nms_scores, nms_classes]

graph = gs.Graph([node],inputs=[boxes, scores], outputs=[nms_num_detections, nms_boxes, nms_scores, nms_classes])

onnx.save(gs.export_onnx(graph),"nms.onnx")



            
            

if __name__ == "__main__":
    model_file = "nms.onnx"
    engine = build_engine(model_file,max_batchsize=batch_size)
    print(engine)

    # engine = load_trt_engine("nms.trt")
    # print(engine)


