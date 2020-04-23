import tensorflow as tf

saved_model_dir = "/Users/jinxiang/Downloads/kfserving-samples/models/tensorflow/flowers/flowers/0001"
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
    summaryWriter = tf.summary.FileWriter('/Users/jinxiang/Downloads/kfserving-samples/models/tensorflow/flowers/flowers/0001', graph)
