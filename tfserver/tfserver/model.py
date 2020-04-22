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

import numpy as np
import tensorflow as tf
import utils as tools

import kfserving


class TFModel(kfserving.KFModel):
    def __init__(self, name, saved_model_dir):
        kfserving.KFModel.__init__(self, name)
        self.name = name
        self.saved_model_dir = saved_model_dir
        self.metadata = tools.getMetadata(saved_model_dir)
        self.graph = tf.Graph()
        self.session_config = None
        self.sess = None
        self.ready = False

    def load(self):
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.allow_soft_placement = True
        self.session_config = config
        with tf.Session(graph=self.graph, config=config) as sess:
            tf.saved_model.loader.load(sess, ['serve'], self.saved_model_dir)
        self.ready = True

    def predict(self, request):
        inputs = []
        try:
            inputs = np.array(request["instances"])
        except Exception as e:
            raise Exception(
                "Failed to initialize Tensorflow Tensor from inputs: %s, %s" % (e, inputs))
        try:
            signature_name = request['signature_name']
            signature_info = self.metadata.signatute_info[signature_name]
            input_tensor = self.graph.get_tensor_by_name(signature_info.input_tensor[0].name)
            output_tensor = self.graph.get_tensor_by_name(signature_info.output_tensor[0].name)
            # output_tensor_names = [t.name for t in signature_info.output_tensor]
        except Exception as e:
            raise Exception("Failed to get signature %s" % e)
        try:
            with tf.Session(graph=self.graph, config=self.session_config) as sess:
                results = sess.run(output_tensor, feed_dict={input_tensor: inputs})
                return {"predictions": results.tolist()}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
    # def load(self):
    #     config = tf.ConfigProto()
    #     config.log_device_placement = True
    #     config.allow_soft_placement = True
    #     self.session_config = config
    #     self.sess = tf.Session(graph=self.graph, config=config)
    #     tf.saved_model.loader.load(self.sess, ['serve'], self.saved_model_dir)
    #     self.ready = True
    #
    # def predict(self, request):
    #     inputs = []
    #     try:
    #         inputs = np.array(request["instances"])
    #     except Exception as e:
    #         raise Exception(
    #             "Failed to initialize Tensorflow Tensor from inputs: %s, %s" % (e, inputs))
    #     try:
    #         signature_name = request['signature_name']
    #         signature_info = self.metadata.signatute_info[signature_name]
    #         input_tensor = self.graph.get_tensor_by_name(signature_info.input_tensor[0].name)
    #         output_tensor = self.graph.get_tensor_by_name(signature_info.output_tensor[0].name)
    #         # output_tensor_names = [t.name for t in signature_info.output_tensor]
    #     except Exception as e:
    #         raise Exception("Failed to get signature %s" % e)
    #     try:
    #         results = self.sess.run(output_tensor, feed_dict={input_tensor: inputs})
    #         return {"predictions": results.tolist()}
    #     except Exception as e:
    #         raise Exception("Failed to predict %s" % e)