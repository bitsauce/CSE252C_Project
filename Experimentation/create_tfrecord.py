import os
import numpy as np
import tensorflow as tf 
from pycocotools.coco import COCO
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

tfrecord_filename = "ms_coco.tfrecord"

def create_record(tfrecord_filename):
    # Create TFRecord writer
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    # Load COCO train data
    dataDir = "../Data/coco"
    dataType = "train2017"
    annFile = "{}/annotations/instances_{}.json".format(dataDir,dataType)

    # Get categories
    coco = COCO(annFile)
    catIds = coco.getCatIds(supNms=["kitchen"])
    #cats = coco.loadCats(catIds)

    print("Load images into memory")
    label_to_cat = {}
    for label, catId in enumerate(catIds):
        img_meta_data = coco.loadImgs(coco.getImgIds(catIds=[catId]))
        for meta_data in img_meta_data[:10]:
            img = io.imread(os.path.join(dataDir, dataType, meta_data["file_name"]))
            img = transform.resize(img, (224, 224))#(480, 640))
            img = img.astype(np.float32)
            feature = {
                "image_raw": bytes_feature(img.tostring()),
                "label": int64_feature(label)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        label_to_cat[label] = catId
    writer.close()
create_record(tfrecord_filename)