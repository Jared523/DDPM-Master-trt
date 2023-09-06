#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda



import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


TRT_LOGGER = trt.Logger()



def get_forward(x,t,stream,engine,context):

    context = engine.create_execution_context()
    # context.set_optimization_profile_async(0, stream.handle)
    print(context.all_binding_shapes_specified)

    context.set_binding_shape(0,x.shape)
    context.set_binding_shape(1,t.shape)
    print(context.all_binding_shapes_specified)
    inputs, outputs, bindings, stream = allocate_buffers(engine,context,stream)
    print('x')
    inputs[0].host=x
    inputs[1].host=t

    trt_outputs=common.do_inference_v2(context,bindings=bindings,inputs=inputs, outputs=outputs, stream=stream)

    return trt_outputs



def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine,context,stream):
    inputs = []
    outputs = []
    bindings = []
    for binding in range(len(engine)):
        size = trt.volume(context.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings , stream


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 24  # 256MiB  1<<30=1G
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model=model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # input_x = network.get_input(0)
            # input_t = network.get_input(1)
            # input_cc = network.get_input(2)
            
            # profile = builder.create_optimization_profile()        
            # profile.set_shape(input_x.name, (6,4,64,64), (6,4,64,64), (6,4,64,64))
            # profile.set_shape(input_t.name, (1,6), (1,6), (1,6))
            # profile.set_shape(input_cc.name, (6,77,768), (6,77,768),(6,77,768))

            # config.add_optimization_profile(profile)



            # print("Completed parsing of ONNX file")
            # print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            # plan = builder.build_serialized_network(network, config)
            # engine = runtime.deserialize_cuda_engine(plan)
            # print("Completed creating Engine")
            network.get_input(0).shape = [1, 3, 64, 64]
            network.get_input(1).shape = [1,1]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")


            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class trt_engine():
    def __init__(self,onnx_path,engine_path):
        
        self.engine=get_engine(onnx_path,engine_path)
        self.context=self.engine.create_execution_context()
        self.stream=cuda.Stream()
    
    def allocate_buffers(self,engine,context,stream):
        inputs = []
        outputs = []
        bindings = []
        for binding in range(len(engine)):
            size = trt.volume(context.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings , stream
    
    def do_inference_v2(self,context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
    def get_forward(self,x,t):

        context = self.engine.create_execution_context()
        stream=self.stream
        # context.set_optimization_profile_async(0, stream.handle)
        print(context.all_binding_shapes_specified)

        context.set_binding_shape(0,x.shape)
        context.set_binding_shape(1,t.shape)

        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine,context,stream)

        inputs[0].host=x
        inputs[1].host=t

        trt_outputs=self.do_inference_v2(context,bindings=bindings,inputs=inputs, outputs=outputs, stream=stream)

        return trt_outputs




def main():
    import time
    # Do inference with TensorRT
    onnx_file_path="/yzpcode/code/stable-diffusion/ddpm_onnx/ddpm.onnx"
    engine_file_path="/yzpcode/code/stable-diffusion/ddpm_onnx/ddpm.trt"
    engine = get_engine(onnx_file_path,engine_file_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()
    x=np.zeros([1,3,64,64],dtype=np.int32)
    t=np.array([[300]],dtype=np.int32)


    for i in  range(10):
        start=time.time()
        trt_out_puts=get_forward(x,t,stream,engine,context)
        outp=trt_out_puts[0].reshape(1,3,64,64)
        end=time.time()
        print("推理时间",end-start)

if __name__ == "__main__":
    main()
