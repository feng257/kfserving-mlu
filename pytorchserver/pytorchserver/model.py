# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import sys

import kfserving
import torch

PYTORCH_FILE = "model.pt"


class PyTorchModel(kfserving.KFModel):
    def __init__(self, name, model_class_name, model_dir):
        kfserving.KFModel.__init__(self, name)
        self.name = name
        self.model_class_name = model_class_name
        self.model_dir = model_dir
        self.ready = False
        self._pytorch = None

    def load(self):
        model_file_dir = kfserving.Storage.download(self.model_dir)
        model_file = os.path.join(model_file_dir, PYTORCH_FILE)
        py_files = []
        for filename in os.listdir(model_file_dir):
            if filename.endswith('.py'):
                py_files.append(filename)
        if len(py_files) == 1:
            model_class_file = os.path.join(model_file_dir, py_files[0])
        elif len(py_files) == 0:
            raise Exception('Missing PyTorch Model Class File.')
        else:
            raise Exception('More than one Python file is detected',
                            'Only one Python file is allowed within model_dir.')
        model_class_name = self.model_class_name

        # Load the python class into memory
        sys.path.append(os.path.dirname(model_class_file))
        modulename = os.path.basename(model_class_file).split('.')[0].replace('-', '_')
        model_class = getattr(importlib.import_module(modulename), model_class_name)

        self._pytorch = model_class()
        self._pytorch.load_state_dict(torch.load(model_file))
        self._pytorch.eval().float().mlu()
        self.ready = True

    def predict(self, request):
        inputs = []
        try:
            inputs = torch.tensor(request["instances"])
        except Exception as e:
            raise Exception(
                "Failed to initialize Torch Tensor from inputs: %s, %s" % (e, inputs))
        try:
            output = self._pytorch(inputs.mlu())
            output = output.cpu().type(torch.FloatTensor)
            return {"predictions":  output.tolist()}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
