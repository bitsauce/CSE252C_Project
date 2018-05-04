import os
import numpy as np
import tensorflow as tf
from resnet import resnet_model
from pycocotools.coco import COCO

# Load COCO train data
dataDir = "../Data/coco"
dataType = "train2017"
annFile = "{}/annotations/instances_{}.json".format(dataDir,dataType)
coco = COCO(annFile)

catIds = coco.getCatIds(supNms=["kitchen"])
cats = coco.loadCats(catIds)
NUM_CLASSES = len(cats)

class ImageNetModel(resnet_model.Model):
    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
                 version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """
        Args:
        resnet_size: The number of convolutional layers needed in the model.
        data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
        num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
        version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
        dtype: The TensorFlow dtype to use for calculations.

        Raises:
        ValueError: if invalid resnet_size is chosen
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(ImageNetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            no_dense=True,
            block_sizes=[3, 4, 6, 3],
            block_strides=[1, 2, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format,
            dtype=dtype
        )

# Define dataset placeholders
batch_size = tf.placeholder(tf.int64)
x_ph = tf.placeholder(tf.float32, [None, None, None, 3], name="x")
y_ph = tf.placeholder(tf.int64, [None], name="y")
is_training_ph = tf.placeholder(tf.bool, name="is_training")

def parse_data(example_proto):
    features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, features)
    
    image = tf.decode_raw(features["image_raw"], tf.float32)
    image = tf.reshape(image, shape=(224, 224, 3), name="REAHDAISJ")#shape=(480, 640, 3))
    image = tf.cast(image, tf.float32)

    label = features["label"]

    return image, label

BATCH_SIZE = 1

# Create dataset and iterator
dataset = tf.data.TFRecordDataset(["ms_coco.tfrecord"])
dataset = dataset.map(parse_data) # Parse the record into tensors
dataset = dataset.batch(BATCH_SIZE)
itr = dataset.make_initializable_iterator()

# Create ResNet50
resnet50 = ImageNetModel(50)

# Feed x through network
x, y = itr.get_next()

tf.summary.image("images", x, max_outputs=6)

inputs = resnet50(x, True)

axes = [2, 3]# if self.data_format == "channels_first" else [1, 2]
inputs = tf.reduce_mean(inputs, axes, keepdims=True)
inputs = tf.identity(inputs, "final_reduce_mean")

inputs = tf.reshape(inputs, [-1, 2048])
inputs = tf.layers.dense(inputs=inputs, units=NUM_CLASSES)
logits = tf.identity(inputs, "final_dense")

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def get_log_dir(model_dir):
    run_idx = 0
    while True:
        log_dir = os.path.join(model_dir, "run_{}".format(run_idx))
        if not os.path.isdir(log_dir):
            return log_dir

log_dir = get_log_dir("ms_coco_classify")
EPOCHS = 10

sess = tf.Session()

average_loss_ph = tf.placeholder(tf.float32)
accuracy_ph   = tf.placeholder(tf.float32)
tf.summary.scalar("loss", average_loss_ph)
tf.summary.scalar("accuracy", accuracy_ph)
merged = tf.summary.merge_all()

# Create file writers
train_summary = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)

# Initialize models vars
sess.run(tf.global_variables_initializer())

# Train loop
for epoch in range(EPOCHS):
    print("Epoch {}/{}: Training...".format(epoch + 1, EPOCHS))

    # Feed train data
    sess.run(itr.initializer)

    # 
    n_iter = 0
    total_loss = 0.0
    total_accuracy = 0.0
    while True:
        try:
            _, loss_value, acc_value = sess.run([train_op, loss, accuracy])
            total_loss += loss_value
            total_accuracy += acc_value
            n_iter += 1
        except tf.errors.OutOfRangeError:
            break
        except Exception as e:
            raise e
    summary = sess.run(merged, feed_dict={ average_loss_ph: total_loss / n_iter, accuracy_ph: total_accuracy / n_iter })
    train_summary.add_summary(summary, epoch)

print("Train done")

train_summary.close()