from resnet import resnet_model
import tensorflow as tf
import os

# TODO:
# - Faster R-CNN
#   - Network
#     - ResNet50 backbone [x]
#     - Anchors
#     - RoIPool
#     - Classification and BB regression heads
#   - Training
#     - Training Loop
#     - MS COCO
#     - Validation
#   - Inference
#
#  - Mask R-CNN
#    - Network
#      - Mask prediction head
#    - Train and test on MS COCO keypoints
#
# - SkeleNet
#   - Predict 3D locations
#   - Integrate constraints

NUM_CLASSES = 10

class ImageNetModel(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

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
        block_sizes=[3, 4, 6, 3],
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        dtype=dtype
    )

input_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
resnet50 = ImageNetModel(50)

output = resnet50(input_ph, True)

"""
axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
inputs = tf.reduce_mean(inputs, axes, keepdims=True)
inputs = tf.identity(inputs, 'final_reduce_mean')

inputs = tf.reshape(inputs, [-1, self.final_size])
inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
inputs = tf.identity(inputs, 'final_dense')
"""

MODEL_NAME = "imagenetv2"

sess = tf.Session()

# Create file writers
train_summary = tf.summary.FileWriter(os.path.join("resnet_test_logs", MODEL_NAME, "train"), sess.graph)

# Initialize models vars
#sess.run(tf.global_variables_initializer())

train_summary.close()