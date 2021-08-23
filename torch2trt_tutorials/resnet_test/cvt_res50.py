#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/18 15:33
# @Author  : wangjianrong
# @File    : cvt_res50.py

from torchvision.models import resnet50
from base.common_tools import init_seed
import torch
from base.myutils import build_engine
from time import time

init_seed(0)

fake_input = torch.randn(4, 3, 224, 224).cuda()

model = resnet50(True).cuda().eval()

# 通过torch2trt转模型
from torch2trt import torch2trt

# 不使用onnx模型
s = time()
trt_model1 = torch2trt(model, [fake_input], max_batch_size=4, fp16_mode=True)
e = time()
print(e - s)
output1 = trt_model1(fake_input)
torch.save(trt_model1.state_dict(),"res50.engine")

# 使用onnx模型
s = time()
trt_model2 = torch2trt(model, [fake_input], max_batch_size=4, fp16_mode=True, use_onnx=True)
e = time()
print(e - s)
output2 = trt_model2(fake_input)

# 对比两个模型的结果
print(torch.max((output1 - output2).abs(), dim=-1))

# 先将torch模型转onnx，然后使用onnx模型转tensorrt
# import onnx

# torch_test.onnx.export(model, fake_input, "res50.onnx")
# engine = build_engine("res50.onnx", max_batchsize=1, fp16_mode=True)
# for binding in engine:
#     origin_shape = engine.get_binding_shape(binding)
#     print(origin_shape)
