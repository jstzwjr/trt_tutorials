import sys

from torch._C import dtype
sys.path.insert(0,"/workspace/projects/trt_tutorials")

from base.myutils import TRTModule,load_trt_engine,TRT_LOGGER,Engine
import torch
import tensorrt as trt
from base.common import allocate_buffers,do_inference_v2
from time import time



class MODEL_INFO:
    model_file = "/workspace/projects/trt_tutorials/nms.trt"
    input_names = ["boxes", "scores"]
    output_names = ["nms_num_detections", "nms_boxes", "nms_scores", "nms_classes"]
    num_classes = 10
    keepTopK = 100
    num_boxes = 2000
    batch_size = 4
    boxes_shape = (batch_size,num_boxes,1,4)
    scores_shape=(batch_size,num_boxes,num_classes)



if __name__ == "__main__":
    # load trt plugins
    # model =TRTModule(MODEL_INFO.model_file,MODEL_INFO.input_names,MODEL_INFO.output_names)
    model = Engine(MODEL_INFO.model_file,MODEL_INFO.input_names,MODEL_INFO.output_names)

    boxes = torch.rand(MODEL_INFO.boxes_shape).float()
    boxes[...,2:4] = boxes[...,:2] + boxes[...,2:4]
    boxes.clip_(0,1)
    boxes = boxes.numpy()
    scores = torch.rand(MODEL_INFO.scores_shape).float()
    scores = scores.numpy()
    
    for _ in range(1000):
        s = time()
        outputs = model(boxes, scores)
        e = time()
        print(e-s)
    # print(outputs[0])
    # print(outputs[3])



