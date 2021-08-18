from time import time
import torch2trt
import torch
import sys

sys.path.append("/workspace/projects/trt_tutorials")
from base.common_tools import init_seed

sys.path.pop(-1)
from torchvision.models import resnet50
import os

# from torch2trt import torch2trt

print(os.environ.get("PYTHONPATH"))

init_seed(0)
fake_input = torch.randn(1, 3, 224, 224).cuda()

torch_model = resnet50(True).cuda().eval()

# convert to TensorRT feeding sample data as input
# model_trt = torch2trt.torch2trt(torch_model, [fake_input], max_batch_size=1, fp16_mode=True)
# torch.save(model_trt.state_dict(), "resnet50.engine")
model_trt = torch2trt.TRTModule()
model_trt.load_state_dict(torch.load("resnet50.engine"))

# torch_model = torch_model.half()
# warm up
for i in range(10):
    y_torch = torch_model(fake_input)

for i in range(10):
    y_trt = model_trt(fake_input)

repeat_time = 1000
s = time()
for _ in range(repeat_time):
    y_torch = torch_model(fake_input)
e = time()
print(e - s)
s = time()
for _ in range(repeat_time):
    y_trt = model_trt(fake_input)
e = time()
print(e - s)

print(torch.max((y_trt - y_torch).abs(), dim=-1))
