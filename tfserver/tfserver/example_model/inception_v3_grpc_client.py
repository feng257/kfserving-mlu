import os
import sys
import threading

import grpc
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
                print("实际类别:" + str(label) + " 预测类别：" + str(prediction))
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


class DataSet(object):
    """Class encompassing test, validation and training MNIST data set."""

    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_imagenet_val_data_sets(data_dir, num_images):
    validation_label_file = os.path.join(data_dir, "validation_label.txt")
    images = []
    labels = []
    with open(validation_label_file, "r") as f:
        validation_label_lines = f.readlines()
    for line in validation_label_lines[0:num_images]:
        arr = line.split(" ")
        image_file = os.path.join(data_dir, "val/" + arr[0])
        label = arr[1]
        image_array = read_tensor_from_image_file(image_file)
        images.append(image_array)
        labels.append(label)
    val_data_set = DataSet(np.array(images), np.array(labels))
    return val_data_set


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=127.5,
                                input_std=127.5):
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
        result = result.reshape([input_height, input_width, 3])
        return result


def grpc_inception_v3_client():
    target = "192.168.113.39:18500"
    num_tests = 4
    concurrency = 4
    # test_data_set = read_imagenet_val_data_sets("/data/imagenet2012/origin_dataset/", num_tests)
    test_data_set = read_imagenet_val_data_sets("/Users/jinxiang/Downloads/imagenet", num_tests)
    result_counter = _ResultCounter(num_tests, concurrency)
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for _ in range(num_tests):
        image, label = test_data_set.next_batch(1)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception_v3'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(image[0], shape=[1, 299, 299, 3]))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label[0], result_counter))
    return result_counter.get_error_rate()


if __name__ == '__main__':
    error_rate = grpc_inception_v3_client()
    print('\nInference error rate: %s%%' % (error_rate * 100))
