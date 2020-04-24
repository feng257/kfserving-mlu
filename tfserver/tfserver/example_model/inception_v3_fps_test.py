from datetime import datetime

import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# 实时计算法 https://www.cnblogs.com/boonya/p/8492287.html
def calc_fps_real(deltaTime):
    '''
    deltaTime : ms
    '''
    fps = (1000.0 / deltaTime)
    return fps


def grpc_inception_v3_client_long_connection():
    target = "192.168.113.39:18500"
    num_tests = 40
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        input_tensor = np.random.normal(-1, 1, [299, 299, 3]).astype(np.float32)
        start = datetime.now()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception_v3'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(input_tensor, shape=[1, 299, 299, 3]))
        result_future = stub.Predict(request, 5.0)  # 5 seconds
        deltaTime = (datetime.now() - start).microseconds
        fps = calc_fps_real(deltaTime)
        print("Real FPS:", fps)


def grpc_mnist_client_long_connection():
    target = "192.168.113.39:28500"
    num_tests = 40
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        input_tensor = np.random.normal(-1, 1, [784]).astype(np.float32)
        start = datetime.now()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(input_tensor, shape=[1, 784]))
        result_future = stub.Predict(request, 5.0)  # 5 seconds
        deltaTime = (datetime.now() - start).microseconds
        fps = calc_fps_real(deltaTime)
        print("Real FPS:", fps)


def grpc_inception_v3_client():
    target = "192.168.113.39:18500"
    num_tests = 40
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        input_tensor = np.random.normal(-1, 1[299, 299, 3])

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception_v3'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(input_tensor, shape=[1, 299, 299, 3]))
        result_future = stub.Predict(request, 5.0)  # 5 seconds


if __name__ == '__main__':
    # grpc_inception_v3_client_long_connection()
    grpc_mnist_client_long_connection()
