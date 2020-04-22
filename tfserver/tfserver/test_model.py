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

import json
import os

import numpy as np
import requests
import tensorflow as tf

from tfserver import TFModel

model_dir = os.path.join(os.path.dirname(__file__), "example_model/inception_v3")


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    with tf.Session() as sess:
        result = sess.run(normalized)
        return result
    return None


def test_model():
    server = TFModel("inception_v3", model_dir, "input", "InceptionV3/Predictions/Reshape_1")
    server.load()

    input_tensor = np.random.normal(-1, 1, [1, 299, 299, 3])

    request = {"instances": input_tensor.tolist()}
    response = server.predict(request)
    print(response["predictions"])


def rest_test_model():
    url = "http://127.0.0.1:8080/v1/models/inception_v3:predict"
    input_tensor = np.random.normal(-1, 1, [1, 299, 299, 3])
    request = {
        "signature_name": "predict_images",
        "instances": input_tensor.tolist()
    }
    response = requests.post(url, data=json.dumps(request))
    print(response.json())

def rest_test_mnist_model():
    url = "http://127.0.0.1:8080/v1/models/mnist:predict"
    input_tensor = np.random.normal(-1, 1, [1, 784])
    request = {
        "signature_name": "predict_images",
        "instances": input_tensor.tolist()
    }
    response = requests.post(url, data=json.dumps(request))
    print(response.json())


def rest_test_model_image(image_file):
    url = "http://127.0.0.1:8080/v1/models/inception_v3:predict"
    input_tensor = read_tensor_from_image_file(image_file, 299, 299, 0, 255)
    request = {"instances": input_tensor.tolist()}
    response = requests.post(url, data=json.dumps(request))
    print(response.json())


def predit_graph_def2():
    model_file = "/Users/jinxiang/Downloads/inception_v3.pb"
    input_name = "input"
    output_name = "InceptionV3/Predictions/Softmax"
    inputs = np.random.normal(-1, 1, [1, 299, 299, 3])
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    input_operation = graph.get_operation_by_name("import/" + input_name)
    output_operation = graph.get_operation_by_name("import/" + output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: inputs
        })
        print(results)


def predit_graph_def():
    model_file = "/Users/jinxiang/Downloads/inception_v3.pb"
    input_tensor_name = "input:0"
    output_tensor_name = "InceptionV3/Predictions/Softmax:0"
    inputs = np.random.normal(-1, 1, [1, 299, 299, 3])
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    input_operation = graph.get_tensor_by_name("import/" + input_tensor_name)
    output_operation = graph.get_tensor_by_name("import/" + output_tensor_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation, {
            input_operation: inputs
        })
        print(results)


def predit_saved_model():
    saved_model_dir = "/Users/jinxiang/Downloads/output_saved_models"
    input_tensor_name = "import/input:0"
    output_tensor_name = "import/InceptionV3/Predictions/Softmax:0"
    inputs = np.random.normal(-1, 1, [1, 299, 299, 3])
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
        input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)
        results = sess.run(output_tensor, feed_dict={input_tensor: inputs})
        print(results)


if __name__ == '__main__':
    # test_model()
    # python -m tfserver --saved_model_dir=/Users/jinxiang/Downloads/output_saved_models --model_name=inception_v3 --workers=1
    # python -m tfserver --saved_model_dir=/Users/jinxiang/Downloads/models/2/1 --model_name=mnist --workers=1
    rest_test_model()
    # rest_test_mnist_model()
    # rest_test_model_image("./example_model/grace_hopper.jpg")
    # predit_graph_def()
