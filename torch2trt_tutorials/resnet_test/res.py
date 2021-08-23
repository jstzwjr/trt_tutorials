#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 10:48
# @Author  : wangjianrong
# @File    : res.py


from torchvision.models import resnet50
import torch
import torch.nn.functional as F


# a = torch.randn((2,10))
# print(a)
# print(F.softmax(a,dim=-1))

import numpy as np
import cv2

def norm(x):
    return x / np.sqrt(np.sum(x**2,axis=-1,keepdims=True))

def norm2(x):
    return x / torch.sqrt(torch.sum(x*x,dim=-1,keepdim=True))



# a = np.random.rand(4,5)
a = torch.rand((4,5))
# b = norm2(a)

# t = x / torch.sqrt(torch.sum(x*x,dim=-1,keepdim=True))
t = norm2(a)
print(t.shape)
print(t)
print(torch.sum(t*t,dim=-1))
# print(t.dtype)
# print(x.shape)
# print(x.dtype)
# x / t

# print(np.sum(b*b,axis=-1))
