import tensorflow as tf

def load_dataset(batch_size):
    # TODO: If record does not exists, create it

    # Data 
    def parse_data(example_proto):
        features = {
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label":     tf.FixedLenFeature([], tf.int64)
        }
        features = tf.parse_single_example(example_proto, features)
        
        image = tf.decode_raw(features["image_raw"], tf.float32)
        image = tf.reshape(image, shape=(224, 224, 3))#shape=(480, 640, 3))
        image = tf.cast(image, tf.float32)

        label = features["label"]

        return image, label

    # Create dataset and iterator
    dataset = tf.data.TFRecordDataset(["ms_coco.tfrecord"])
    dataset = dataset.map(parse_data) # Parse the record into tensors
    dataset = dataset.batch(batch_size)
    return dataset
