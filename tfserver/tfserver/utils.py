from tensorflow.core.framework import types_pb2
from tensorflow.python.tools import saved_model_utils


class ModelMetadata():
    def __init__(self, saved_model_dir):
        self.saved_model_dir = saved_model_dir
        self.tag_set = 'serve'
        self.signatute_info = {}


class SignatureInfo():
    def __init__(self, signature_name):
        self.signature_name = signature_name
        self.input_tensor = []
        self.output_tensor = []
        self.method_name = ""


class TensorInfo():
    def __init__(self, name, dtype, shape):
        self.name = name
        self.shape = shape
        self.dtype = dtype


def getMetadata(saved_model_dir):
    model_metadata = ModelMetadata(saved_model_dir)
    signature_def_map = saved_model_utils.get_meta_graph_def(saved_model_dir, model_metadata.tag_set).signature_def
    for signature_def_key in sorted(signature_def_map.keys()):
        signature_info = SignatureInfo(signature_def_key)
        meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, model_metadata.tag_set)
        inputs_tensor_info = meta_graph_def.signature_def[signature_def_key].inputs
        outputs_tensor_info = meta_graph_def.signature_def[signature_def_key].outputs
        method_name = meta_graph_def.signature_def[signature_def_key].method_name
        for input_key, input_tensor in sorted(inputs_tensor_info.items()):
            name, dtype, shape = _get_tensor_info(input_tensor)
            signature_info.input_tensor.append(TensorInfo(name, dtype, shape))
        for output_key, output_tensor in sorted(outputs_tensor_info.items()):
            name, dtype, shape = _get_tensor_info(output_tensor)
            signature_info.output_tensor.append(TensorInfo(name, dtype, shape))
        signature_info.method_name = method_name
        model_metadata.signatute_info[signature_def_key] = signature_info
    return model_metadata

def _get_tensor_info(tensor_info):
    dtype = {value: key for (key, value) in types_pb2.DataType.items()}[tensor_info.dtype]
    # Display shape as tuple.
    if tensor_info.tensor_shape.unknown_rank:
        shape = 'unknown_rank'
    else:
        dims = [dim.size for dim in tensor_info.tensor_shape.dim]
        shape = tuple(dims)
    return tensor_info.name, dtype, shape

def flatten(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            for bar in flatten(k):
                yield bar
if __name__ == '__main__':
    saved_model_dir = "/Users/jinxiang/Downloads/models/efficientNet_b3"
    data = getMetadata(saved_model_dir)
    import json

    print(json.dumps(data, default=lambda obj: obj.__dict__))