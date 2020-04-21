# This is a placeholder for a Google-internal import.

import argparse
import tensorflow as tf


def pb_to_saved_model():
    builder = tf.saved_model.builder.SavedModelBuilder(args.export_dir)
    with tf.gfile.GFile(args.graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def)
        input_ids = sess.graph.get_operation_by_name("import/" + args.input_layer).outputs[0]
        output_ids = sess.graph.get_operation_by_name("import/" + args.output_layer).outputs[0]
        tensor_info_input = tf.saved_model.utils.build_tensor_info(input_ids)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(output_ids)
        prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_input},
            outputs={'scores': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            })
    builder.save()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="graph/model to be executed")
    parser.add_argument("--export_dir", required=True, help="export saved model dir")
    parser.add_argument("--input_layer", required=True, help="name of input layer")
    parser.add_argument("--output_layer", required=True, help="name of output layer")
    args = parser.parse_args()
    pb_to_saved_model()
    '''
    python saved_model_tools.py 
        --graph=./inception_v3.pb \
        --export_dir=/Users/jinxiang/Downloads/output_saved_models \
        --input_layer=input \
        --output_layer=InceptionV3/Predictions/Softmax
    '''
