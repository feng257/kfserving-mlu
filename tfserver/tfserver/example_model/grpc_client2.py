import sys
import threading

import grpc
import mnist_input_data
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time

class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests):
        self._num_tests = num_tests
        self._error = 0
        self._done = 0
        self._active = 0

    def inc_error(self):
        self._error += 1

    def inc_done(self):
        self._done += 1

    def dec_active(self):
        self._active -= 1

    def get_error_rate(self):
        while self._done != self._num_tests:
            time.sleep(0.1)
        return self._error / float(self._num_tests)


def _create_rpc_callback(label, result_counter):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = np.array(
                result_future.result().outputs['scores'].float_val)
            prediction = np.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


def grpc_mnist_client():
    target = "192.168.113.39:18500"
    num_tests = 100
    test_data_set = mnist_input_data.read_data_sets("/Users/jinxiang/Downloads/models/2/data").test
    result_counter = _ResultCounter(num_tests)
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        image, label = test_data_set.next_batch(1)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(image[0], shape=[1, 784]))
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label[0], result_counter))
    return result_counter.get_error_rate()


def grpc_mnist_client2():
    target = "192.168.113.39:18500"
    inputs = np.random.normal(0, 1, [784]).astype(np.float32)

    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(tf.make_tensor_proto(inputs, shape=[1, 784]))
    result = stub.Predict(request, 5.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
    error_rate = grpc_mnist_client()
    print('\nInference error rate: %s%%' % (error_rate * 100))
    # grpc_mnist_client2()
