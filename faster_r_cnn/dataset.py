import tensorflow as tf

def parse_record_fn(example_proto, is_training):
    features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, features)
    
    image = tf.decode_raw(features["image_raw"], tf.float32)
    image = tf.reshape(image, shape=(64, 64, 3))
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    label = features["label"]
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(tf.reshape(label, shape=[]), 10)

    return image, label

def get_dataset_size(tfrecord_file):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_file))