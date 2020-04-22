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
        self.metadata = None
        self.session_config = None
        self.sess = None
        self.ready = False

    def load(self):
        model_file_dir = kfserving.Storage.download(self.saved_model_dir)
        self.metadata = tools.getMetadata(model_file_dir)
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True
        self.session_config = config
        self.sess = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(self.sess, ['serve'], model_file_dir)
        tf.saved_model.load
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
            input_tensor = self.sess.graph.get_tensor_by_name(signature_info.input_tensor[0].name)
            output_tensors = self.sess.graph.get_tensor_by_name(signature_info.output_tensor[0].name)
        except Exception as e:
            raise Exception("Failed to get signature %s" % e)
        try:
            results = self.sess.run(output_tensors, feed_dict={input_tensor: inputs})
            if len(results) > 1:
                results = [arr.tolist() for arr in results]
            else:
                results = results[0].tolist()
            return {"predictions": results}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
