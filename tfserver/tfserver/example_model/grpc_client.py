import sys
import threading

import grpc
import mnist_input_data
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """

    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
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
    concurrency = 4
    test_data_set = mnist_input_data.read_data_sets("/Users/jinxiang/Downloads/models/2/data").test
    result_counter = _ResultCounter(num_tests, concurrency)
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        image, label = test_data_set.next_batch(1)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(image[0], shape=[1, 784]))
        result_counter.throttle()
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
