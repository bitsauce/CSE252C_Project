import os
import numpy as np
import tensorflow as tf 
import skimage.io as io
import skimage.transform as transform
from pycocotools.coco import COCO

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

tfrecord_file_prefix = "ms_coco_bbox"

def create_record(tfrecord_filename):
    for phase in ["train", "val"]:
        # Load train data
        data_dir  = "../data/coco"
        data_type = "{}2017".format(phase)
        ann_file  = "{}/annotations/instances_{}.json".format(data_dir, data_type)

        # Initialize COCO api for instance annotations
        coco = COCO(ann_file)

        # Get image ids for the 10 first categories
        imgIds = set()
        catIds = coco.getCatIds()[:10]
        for catId in catIds:
            for imgId in coco.getImgIds(catIds=[catId])[:700]:
                imgIds.add(imgId)
        imgIds = list(imgIds)

        # Create TFRecord writer
        print("Writing {} images to {}_{}.tfrecord".format(len(imgIds), tfrecord_file_prefix, phase))
        writer = tf.python_io.TFRecordWriter("{}_{}.tfrecord".format(tfrecord_file_prefix, phase))

        # Save every image
        for imgId in imgIds:
            # load and display image
            image_file = coco.loadImgs(imgId)[0]["file_name"]
            image = io.imread(os.path.join(data_dir, data_type, image_file)) / 255.0

            # Turn greyscale images into RGB
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)

            # Get bounding boxes
            annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
            anns = coco.loadAnns(annIds)
            bboxes = []
            labels = []
            for ann in anns:
                if ann["category_id"] > 10: continue
                bboxes.append(ann["bbox"])
                labels.append(ann["category_id"])
                
            if len(bboxes) == 0:
                print("Missing category?")
                print(anns)
                continue

            bboxes = np.array(bboxes, dtype=np.int64)

            # Convert to float
            image = image.astype(np.float32)
            feature = {
                "image/width": int64_feature(image.shape[1]),
                "image/height": int64_feature(image.shape[0]),
                "image/data": bytes_feature(image.tostring()),
                "bbox/xmin": int64_list_feature(bboxes[:, 0]),
                "bbox/ymin": int64_list_feature(bboxes[:, 1]),
                "bbox/xmax": int64_list_feature(bboxes[:, 2]),
                "bbox/ymax": int64_list_feature(bboxes[:, 3]),
                "bbox/label": int64_list_feature(labels)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

create_record(tfrecord_file_prefix)