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
        result = result.reshape(input_width, input_height, 3)
        return result
    return None


def test_model():
    server = TFModel("inception_v3", model_dir, "input", "InceptionV3/Predictions/Reshape_1")
    server.load()

    input_tensor = np.random.normal(-1, 1, [299, 299, 3])

    request = {"instances": input_tensor.tolist()}
    response = server.predict(request)
    print(response["predictions"])


def rest_test_model():
    url = "http://127.0.0.1:8080/v1/models/inception_v3:predict"
    input_tensor = np.random.normal(-1, 1, [299, 299, 3])
    request = {"instances": input_tensor.tolist()}
    response = requests.post(url, data=json.dumps(request))
    print(response.json())


def rest_test_model_image(image_file):
    url = "http://127.0.0.1:8080/v1/models/inception_v3:predict"
    input_tensor = read_tensor_from_image_file(image_file, 299, 299, 0, 255)
    request = {"instances": input_tensor.tolist()}
    response = requests.post(url, data=json.dumps(request))
    print(response.json())


if __name__ == '__main__':
    # test_model()
    # python -m tfserver --model_dir=example_model/inception_v3 --model_name=inception_v3  --input_name=import/input --output_name=import/InceptionV3/Predictions/Reshape_1
    rest_test_model()
    rest_test_model_image("./example_model/grace_hopper.jpg")
