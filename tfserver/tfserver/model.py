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

import os

import kfserving
import numpy as np
import tensorflow as tf

import utils as tools

TENSORFLOW_MODEL_FILE = "saved_model.pb"


class TFModel(kfserving.KFModel):
    def __init__(self, name, saved_model_dir, input_name, output_name):
        kfserving.KFModel.__init__(self,name)
        self.name = name
        self.saved_model_dir = saved_model_dir
        self.input_name = input_name
        self.output_name = output_name
        self.ready = False
        self.graph = None
        self._batch_size = 1

    def load(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        model_file = os.path.join(self.saved_model_dir, TENSORFLOW_MODEL_FILE)
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        self.graph = graph
        self.ready = True

    def predict(self, request):
        inputs = []
        try:
            inputs = np.array(request["instances"])
            input_shape = tuple(tools.flatten([(self._batch_size), inputs.shape]))
            inputs = inputs.reshape(input_shape)
        except Exception as e:
            raise Exception(
                "Failed to initialize Tensorflow Tensor from inputs: %s, %s" % (e, inputs))
        try:
            input_operation = self.graph.get_operation_by_name("import/" + self.input_name)
            output_operation = self.graph.get_operation_by_name("import/" + self.output_name)
        except Exception as e:
            raise Exception("Failed to get signature %s" % e)
        try:
            with tf.Session(graph=self.graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: inputs
                })
            return {"predictions": results.tolist()}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
