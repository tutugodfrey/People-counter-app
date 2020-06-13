#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, args):
        ### TODO: Initialize any class variables desired ###
        self.model_xml = args.model
        self.cpu_extension = args.cpu_extension
        self.plugin = None
        self.net_plugin = None
        self.input_blob = None
        self.output_blob = None
        self.infer_request_handle = None
        self.exec_network = None
        self.infer_request = None
        self.network = None

    def load_model(self):
        ### TODO: Load the model ###
        self.plugin = IECore()
        model_bin = '.'.join([os.path.splitext(self.model_xml)[0], 'bin'])
#         self.net_plugin = IENetwork(model_xml, model_bin)
        self.network = IENetwork(self.model_xml, model_bin)
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.network, device_name='CPU')
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if unsupported_layers:
            print('Unsupported layers {}'.format(unsupported_layers))
            print('Please check whether extension are available to add to IECore')
#             exit(1)

        ### TODO: Add any necessary extensions ###
#         CPU_EXTENSION = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
        self.plugin.add_extension(self.cpu_extension, 'CPU')
#         self.plugin.add_extension(CPU_EXTENSION, 'CPU')

        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(network=self.network, device_name='CPU', num_requests=1)
        
        ### Note: You may need to update the function parameters. ###
        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###)
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        print(self.input_blob, self.output_blob)
        input_shape = self.network.inputs[self.input_blob].shape
        return input_shape

    def exec_net(self, frame):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs = {self.input_blob:frame})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
