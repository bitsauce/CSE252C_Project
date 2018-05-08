import os
import numpy as np
import tensorflow as tf 
import skimage.io as io
import skimage.transform as transform

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

tfrecord_file_prefix = "tiny_imagenet_1"

print("Creating", tfrecord_file_prefix)

def create_record(tfrecord_filename):
    # Load train data
    data_dir = "../data/tiny_imagenet"

    class_id_to_name = {}
    class_name_to_id = {}
    with open(os.path.join(data_dir, "wnids.txt"), "r") as f:
        for i, line in enumerate(f.readlines()):
            name = line.strip()
            class_id_to_name[i] = name
            class_name_to_id[name] = i

    # Create TFRecord writer
    writer = tf.python_io.TFRecordWriter("{}_train.tfrecord".format(tfrecord_file_prefix))
    for i in range(10):
        print("Processing class %i"%i)
        image_dir = os.path.join(data_dir, "train", class_id_to_name[i], "images")
        for image_file in list(os.listdir(image_dir))[:10]:
            image = io.imread(os.path.join(image_dir, image_file)) / 255.0
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)

            if len(image.shape) != 3 or image.shape[0] != 64 or image.shape[1] != 64 or image.shape[2] != 3:
                print(image.shape)
                exit(-1)
                
            image = image.astype(np.float32)
            feature = {
                "image_raw": bytes_feature(image.tostring()),
                "label": int64_feature(i)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # Close writer
    writer.close()

    # Create TFRecord writer
    writer = tf.python_io.TFRecordWriter("{}_val.tfrecord".format(tfrecord_file_prefix))
    with open(os.path.join(data_dir, "val", "val_annotations.txt"), "r") as f:
        for line in f.readlines():
            ann = line.strip().split()
            file_name = ann[0]
            class_name = ann[1]
            class_id = class_name_to_id[class_name]
            if class_id != i:
                continue

            image_file = os.path.join(data_dir, "val", "images", file_name)
            image = io.imread(image_file) / 255.0

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)

            if len(image.shape) != 3 or image.shape[0] != 64 or image.shape[1] != 64 or image.shape[2] != 3:
                print(image.shape)
                exit(-1)

            image = image.astype(np.float32)
            feature = {
                "image_raw": bytes_feature(image.tostring()),
                "label": int64_feature(class_id)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # Close writer
    writer.close()
create_record(tfrecord_file_prefix)