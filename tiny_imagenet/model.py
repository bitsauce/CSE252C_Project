import resnet
import tensorflow as tf

def get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

class ResNetClassifier(resnet.Model):
    def __init__(self, resnet_size, num_classes, data_format=None,
                 resnet_version=resnet.DEFAULT_VERSION,
                 dtype=resnet.DEFAULT_DTYPE):
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
        else:
            bottleneck = True

        super(ResNetClassifier, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            fully_conv=True,
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )
        
        if resnet_size < 50:
            self.final_size = 512
        else:
            self.final_size = 2048

    def __call__(self, inputs, training):
        inputs = super(ResNetClassifier, self).__call__(inputs, training)

        with self._model_variable_scope():
            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.reshape(inputs, [-1, self.final_size])
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')

        return inputs

