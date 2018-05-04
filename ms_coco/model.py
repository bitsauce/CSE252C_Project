from resnet import resnet_model
import tensorflow as tf

from pycocotools.coco import COCO

# Load COCO train data
dataDir = "../Data/coco"
dataType = "train2017"
annFile = "{}/annotations/instances_{}.json".format(dataDir,dataType)
coco = COCO(annFile)
catIds = coco.getCatIds(supNms=["kitchen"])
cats = coco.loadCats(catIds)
NUM_CLASSES = len(cats)

class MSCocoModel(resnet_model.Model):
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

        super(MSCocoModel, self).__init__(
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

    def __call__(self, inputs, training):
        inputs = resnet_model.Model.__call__(self, inputs, training)
        
        axes = [2, 3] if self.data_format == "channels_first" else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
        inputs = tf.identity(inputs, "final_reduce_mean")

        inputs = tf.reshape(inputs, [-1, 2048])
        inputs = tf.layers.dense(inputs=inputs, units=NUM_CLASSES)
        logits = tf.identity(inputs, "final_dense")

        return logits

