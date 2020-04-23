import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2
import google.protobuf.json_format as json_format
import numpy as np
import onnxruntime
import requests
from onnxruntime.datasets import get_example


# onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz"

def predict_inception_v3():
    # https://raw.githubusercontent.com/ChangweiZhang/Awesome-ONNX-Models/master/Inceptionv3.onnx
    example_model = get_example("/Users/jinxiang/Downloads/models/inception_v3.onnx")
    sess = onnxruntime.InferenceSession(example_model)

    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[0].type
    print("Input type  :", input_type)

    output_names = [t.name for t in sess.get_outputs()]
    print("Output name  :", output_names)
    output_shapes = [t.shape for t in sess.get_outputs()]
    print("Output shape :", output_shapes)
    output_type = sess.get_outputs()[0].type
    print("Output type  :", output_type)

    x = np.random.normal(-1, 1, [1, 3, 299, 299])
    if input_type == "tensor(float16)":
        x = x.astype(np.float16)
    elif input_type == "tensor(float32)":
        x = x.astype(np.float32)
    elif input_type == "tensor(float64)":
        x = x.astype(np.float64)
    result = sess.run(output_names, {input_name: x})
    print(result)


def predict_inception_v1():
    # https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz
    example_model = get_example("/Users/jinxiang/Downloads/inception_v1/model.onnx")
    sess = onnxruntime.InferenceSession(example_model)

    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[0].type
    print("Input type  :", input_type)
    output_names = [t.name for t in sess.get_outputs()]
    print("Output name  :", output_names)
    output_shapes = [t.shape for t in sess.get_outputs()]
    print("Output shape :", output_shapes)
    output_type = sess.get_outputs()[0].type
    print("Output type  :", output_type)

    x = np.random.normal(-1, 1, [1, 3, 224, 224])
    if input_type == "tensor(float)":
        x = x.astype(np.float32)
    elif input_type == "tensor(float16)":
        x = x.astype(np.float16)
    elif input_type == "tensor(float32)":
        x = x.astype(np.float32)
    elif input_type == "tensor(float64)":
        x = x.astype(np.float64)
    result = sess.run(output_names, {input_name: x})
    print(result)


def predict_inception_v1():
    # https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz
    example_model = get_example("/Users/jinxiang/Downloads/inception_v1/model.onnx")
    sess = onnxruntime.InferenceSession(example_model)

    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[0].type
    print("Input type  :", input_type)
    output_names = [t.name for t in sess.get_outputs()]
    print("Output name  :", output_names)
    output_shapes = [t.shape for t in sess.get_outputs()]
    print("Output shape :", output_shapes)
    output_type = sess.get_outputs()[0].type
    print("Output type  :", output_type)

    x = np.random.normal(-1, 1, [1, 3, 224, 224]).astype(np.float32)
    result = sess.run(output_names, {input_name: x})
    print(result)


def predict_inception_v3_cpu():
    # https://raw.githubusercontent.com/ChangweiZhang/Awesome-ONNX-Models/master/Inceptionv3.onnx
    example_model = get_example("/Users/jinxiang/Downloads/models/inception_v3.onnx")
    sess = onnxruntime.InferenceSession(example_model)
    input_name = sess.get_inputs()[0].name
    input_type = sess.get_inputs()[0].type
    output_names = [t.name for t in sess.get_outputs()]
    inputs = np.random.normal(-1, 1, [1, 3, 299, 299])
    if input_type == "tensor(float16)":
        inputs = inputs.astype(np.float16)
    elif input_type == "tensor(float32)":
        inputs = inputs.astype(np.float32)
    result = sess.run(output_names, {input_name: inputs})
    print(result)


def rest_protobuf_inception_v1_test():
    inputs = np.random.normal(-1, 1, [1, 3, 224, 224]).astype(np.float32)
    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(inputs.shape)
    input_tensor.data_type = 1
    input_tensor.raw_data = inputs.tobytes()

    request_message = predict_pb2.PredictRequest()

    # For your model, the inputs name should be something else customized by yourself. Use Netron to find out the input name.
    request_message.inputs["data_0"].data_type = input_tensor.data_type
    request_message.inputs["data_0"].dims.extend(input_tensor.dims)
    request_message.inputs["data_0"].raw_data = input_tensor.raw_data

    content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

    request_headers = {
        'Content-Type': 'application/vnd.google.protobuf',
        'Accept': 'application/x-protobuf'
    }
    inference_url = "http://192.168.113.39:8002/v1/models/default/versions/1:predict"
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)
    output = np.frombuffer(response_message.outputs['prob_1'].raw_data, dtype=np.float32)
    print(output)


def rest_http_inception_v1_test():
    inputs = np.random.normal(-1, 1, [1, 3, 224, 224]).astype(np.float32)
    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(inputs.shape)
    input_tensor.data_type = 1
    input_tensor.raw_data = inputs.tobytes()
    request_message = predict_pb2.PredictRequest()

    # For your model, the inputs name should be something else customized by yourself. Use Netron to find out the input name.
    request_message.inputs["data_0"].data_type = input_tensor.data_type
    request_message.inputs["data_0"].dims.extend(input_tensor.dims)
    request_message.inputs["data_0"].raw_data = input_tensor.raw_data
    # write message data to JSON
    message_data = json_format.MessageToJson(request_message)
    # Call predictor
    inference_url = "http://192.168.113.39:8002/v1/models/default/versions/1:predict"

    request_headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = requests.post(inference_url, headers=request_headers, data=message_data)

    # Parse response message
    response_message = json_format.Parse(response.text, predict_pb2.PredictResponse())
    output1 = np.frombuffer(response_message.outputs['prob_1'].raw_data, dtype=np.float32)
    print(output1)


def rest_ssd_test():
    import numpy as np
    import assets.onnx_ml_pb2 as onnx_ml_pb2
    import assets.predict_pb2 as predict_pb2
    import requests
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Load the raw image

    input_shape = (1, 3, 1200, 1200)
    img = Image.open("assets/blueangels.jpg")
    img = img.resize((1200, 1200), Image.BILINEAR)
    # Preprocess and normalize the image

    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    # Create request message to be sent to the ORT server

    input_tensor = onnx_ml_pb2.TensorProto()
    print(norm_img_data.shape)
    input_tensor.dims.extend(norm_img_data.shape)
    input_tensor.data_type = 1
    input_tensor.raw_data = norm_img_data.tobytes()

    request_message = predict_pb2.PredictRequest()

    # For your model, the inputs name should be something else customized by yourself. Use Netron to find out the input name.
    request_message.inputs["image"].data_type = input_tensor.data_type
    request_message.inputs["image"].dims.extend(input_tensor.dims)
    request_message.inputs["image"].raw_data = input_tensor.raw_data

    content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

    for h in content_type_headers:
        request_headers = {
            'Content-Type': h,
            'Accept': 'application/x-protobuf'
        }
    # Parse response message
    inference_url = "http://192.168.113.39:8001/v1/models/default/versions/1:predict"
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)

    # For your model, the outputs names should be something else customized by yourself. Use Netron to find out the outputs names.
    bboxes = np.frombuffer(response_message.outputs['bboxes'].raw_data, dtype=np.float32)
    labels = np.frombuffer(response_message.outputs['labels'].raw_data, dtype=np.int64)
    scores = np.frombuffer(response_message.outputs['scores'].raw_data, dtype=np.float32)

    print('Boxes shape:', response_message.outputs['bboxes'].dims)
    print('Labels shape:', response_message.outputs['labels'].dims)
    print('Scores shape:', response_message.outputs['scores'].dims)
    ## Display image with bounding boxes and appropriate class

    # Parse the list of class labels
    classes = [line.rstrip('\n') for line in open('assets/coco_classes.txt')]

    # Plot the bounding boxes on the image
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    resized_width = 1200  # we resized the original image, remember ?
    resized_height = 1200
    num_boxes = 6  # we limit displaying to just 10 boxes to avoid clogging the result image with boxes
    # The results are already sorted based on box confidences, so we just pick top N boxes without sorting

    for c in range(num_boxes):
        base_index = c * 4
        y1, x1, y2, x2 = bboxes[base_index] * resized_height, bboxes[base_index + 1] * resized_width, bboxes[
            base_index + 2] * resized_height, bboxes[base_index + 3] * resized_width
        color = 'blue'
        box_h = (y2 - y1)
        box_w = (x2 - x1)
        bbox = patches.Rectangle((y1, x1), box_h, box_w, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(y1, x1, s=classes[labels[c] - 1], color='white', verticalalignment='top',
                 bbox={'color': color, 'pad': 0})
    plt.axis('off')

    # Save image
    plt.savefig("assets/ssd_result.jpg", bbox_inches='tight', pad_inches=0.0)
    plt.show()


def rest_ssd_norm_test():
    inference_url = "http://192.168.113.39:8001/v1/models/default/versions/1:predict"
    inputs = np.random.normal(-1, 1, [1, 3, 1200, 1200]).astype(np.float32)
    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(inputs.shape)
    input_tensor.data_type = 1
    input_tensor.raw_data = inputs.tobytes()
    request_message = predict_pb2.PredictRequest()
    # For your model, the outputs names should be something else customized by yourself.
    # Use Netron to find out the outputs names.
    request_message.inputs["image"].data_type = input_tensor.data_type
    request_message.inputs["image"].dims.extend(input_tensor.dims)
    request_message.inputs["image"].raw_data = input_tensor.raw_data
    request_headers = {
        'Content-Type': 'application/vnd.google.protobuf',
        'Accept': 'application/x-protobuf'
    }
    # Parse response message
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)
    # For your model, the outputs names should be something else customized by yourself.
    # Use Netron to find out the outputs names.
    bboxes = np.frombuffer(response_message.outputs['bboxes'].raw_data, dtype=np.float32)
    labels = np.frombuffer(response_message.outputs['labels'].raw_data, dtype=np.int64)
    scores = np.frombuffer(response_message.outputs['scores'].raw_data, dtype=np.float32)
    print(bboxes,labels,scores)


def rest_inception_v3_test():
    inputs = np.random.normal(-1, 1, [1, 3, 299, 299]).astype(np.float16)
    input_tensor = onnx_ml_pb2.TensorProto()
    input_tensor.dims.extend(inputs.shape)
    input_tensor.data_type = 10
    input_tensor.raw_data = inputs.tobytes()
    request_message = predict_pb2.PredictRequest()

    # For your model, the inputs name should be something else customized by yourself. Use Netron to find out the input name.
    request_message.inputs["image"].data_type = input_tensor.data_type
    request_message.inputs["image"].dims.extend(input_tensor.dims)
    request_message.inputs["image"].raw_data = input_tensor.raw_data

    content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

    for h in content_type_headers:
        request_headers = {
            'Content-Type': h,
            'Accept': 'application/x-protobuf'
        }
    # Parse response message
    inference_url = "http://192.168.113.39:8001/v1/models/default/versions/1:predict"
    response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
    response_message = predict_pb2.PredictResponse()
    response_message.ParseFromString(response.content)


if __name__ == '__main__':
    # predict_inception_v3()
    # rest_http_inception_v1_test()
    # rest_ssd_test()
    # predict_inception_v3_cpu()
    # predict_inception_v1()
    rest_ssd_norm_test()
