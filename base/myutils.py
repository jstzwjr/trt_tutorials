
import tensorrt as trt
from base import common
import os
import torch
import pycuda.driver as cuda
from base.common import allocate_buffers,do_inference_v2
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def build_engine(onnx_file,trt_file=None,save_engine=True,max_batchsize=1,fp16_mode=False,strict_type_constraints=False):
    '''
    model_file: onnx model path
    '''
    with trt.Builder(TRT_LOGGER) as builder,builder.create_network(common.EXPLICIT_BATCH) as network,trt.OnnxParser(network,TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = max_batchsize
        builder.fp16_mode = fp16_mode
        builder.strict_type_constraints = strict_type_constraints
        if not parser.parse_from_file(onnx_file):
            print("parse model failed!")
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            return None

        with builder.build_cuda_engine(network) as engine:
            if save_engine:
                if trt_file is None:
                    trt_file = os.path.splitext(onnx_file)[0] + '.trt'

                with open(trt_file,"wb") as f:
                    f.write(engine.serialize())
            return engine

def load_trt_engine(trt_file):
    # load trt engine
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(trt_file,'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)

def np_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return np.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return np.bool
    elif dtype == trt.int32:
        return np.int32
    elif dtype == trt.float16:
        return np.float16
    elif dtype == trt.float32:
        return np.float32
    else:
        raise TypeError("%s is not supported by numpy" % dtype)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class Engine:
    def __init__(self, engine_path, input_names, output_names):
        # load engine
        trt.init_libnvinfer_plugins(None, "")
        self.engine = load_trt_engine(engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        
        # save input and output shapes and types
        self.input_names = input_names
        self.output_names = output_names
        self.input_shapes = []
        self.input_types = []
        self.output_shapes = []
        self.output_types = []
        for input_name in self.input_names:
            idx = self.engine.get_binding_index(input_name)
            input_dtype = np_dtype_from_trt(self.engine.get_binding_dtype(idx))
            input_shape = self.engine.get_binding_shape(idx)
            self.input_shapes.append(input_shape)
            self.input_types.append(input_dtype)

        for output_name in self.output_names:
            idx = self.engine.get_binding_index(output_name)
            output_dtype = self.engine.get_binding_dtype(idx)
            output_shape = self.engine.get_binding_shape(idx)
            self.output_shapes.append(output_shape)
            self.output_types.append(output_dtype)
        print(self.input_shapes)
        print(self.output_shapes)

    def __call__(self,*inputs):
        for i in range(len(self.input_names)):
            self.inputs[i].host = inputs[i].astype(self.input_types[i])
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        results = []
        for i in range(len(self.output_names)):
            results.append(trt_outputs[i].reshape(self.output_shapes[i]))
        return results


# 测试结果不正确
class TRTModule(torch.nn.Module):
    def __init__(self, engine_path=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)

        trt.init_libnvinfer_plugins(None, "")
        self.engine = load_trt_engine(engine_path)

        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            print(idx,self.engine.get_binding_shape(idx))
            # shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
            shape = tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()

        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )
        # self.context.execute_async_v2(bindings=bindings, stream_handle=cuda.Stream().handle)

        outputs = tuple(outputs)
        for output in outputs:
            print(output.shape)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()