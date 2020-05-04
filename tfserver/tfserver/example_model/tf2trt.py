import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = "/Users/jinxiang/Downloads/output_saved_models/1"
output_saved_model_dir = "/Users/jinxiang/Downloads/output_saved_models/trt"

converter = trt.TrtGraphConverter(
    input_saved_model_dir=input_saved_model_dir,
    max_workspace_size_bytes=(11 < 32),
    precision_mode="FP16",
    is_dynamic_op=True,
    maximum_cached_engines=100)
converter.convert()
converter.save(output_saved_model_dir)

# with tf.Session() as sess:
#     # First load the SavedModel into the session
#     tf.saved_model.loader.load(
#         sess, [tf.saved_model.tag_constants.SERVING],
#        output_saved_model_dir)
#     output = sess.run([output_tensor], feed_dict={input_tensor: input_data})
