import os
import numpy as np
import tensorflow as tf
from model import ResNetClassifier
from dataset import parse_record_fn, get_dataset_size
from utils import get_log_dir, add_scalar_summary

import resnet_run_loop

class FasterRCNNTrainer:
    def __init__(self):
        # Initialize network
        #self.model = ResNetClassifier(resnet_size=50, num_classes=10)
        pass

    def model_fn(self, weight_decay=1e-4, momentum=0.9, loss_filter_fn=None, learning_rate_fn=None, dtype=tf.float32):
        def func(features, labels, mode):
            # Generate a summary node for the images
            tf.summary.image("images", features, max_outputs=6)

            features = tf.cast(features, dtype)

            model = ResNetClassifier(resnet_size=50, num_classes=10, dtype=dtype)
            logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

            # This acts as a no-op if the logits are already in fp32 (provided logits are
            # not a SparseTensor). If dtype is is low precision, logits must be cast to
            # fp32 for numerical stability.
            logits = tf.cast(logits, tf.float32)

            predictions = {
                "classes": tf.argmax(logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                # Return the predictions and the specification for serving a SavedModel
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={
                        "predict": tf.estimator.export.PredictOutput(predictions)
                    })

            # Calculate loss, which includes softmax cross entropy and L2 regularization.
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)

            # Create a tensor named cross_entropy for logging purposes.
            tf.identity(cross_entropy, name="cross_entropy")
            tf.summary.scalar("cross_entropy", cross_entropy)

            # If no loss_filter_fn is passed, assume we want the default behavior,
            # which is that batch_normalization variables are excluded from loss.
            def exclude_batch_norm(name):
                return "batch_normalization" not in name
            nonlocal loss_filter_fn
            loss_filter_fn = loss_filter_fn or exclude_batch_norm

            # Add weight decay to the loss.
            l2_loss = weight_decay * tf.add_n(
                # loss is computed using fp32 for numerical stability.
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                if loss_filter_fn(v.name)])
            tf.summary.scalar("l2_loss", l2_loss)
            loss = cross_entropy + l2_loss

            if mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()

                learning_rate = learning_rate_fn(global_step)

                # Create a tensor named learning_rate for logging purposes
                tf.identity(learning_rate, name="learning_rate")
                tf.summary.scalar("learning_rate", learning_rate)

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate#,
                    #momentum=momentum
                )
                
                loss_scale = 1
                if loss_scale != 1:
                    # When computing fp16 gradients, often intermediate tensor values are
                    # so small, they underflow to 0. To avoid this, we multiply the loss by
                    # loss_scale to make these tensor values loss_scale times bigger.
                    scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

                    # Once the gradient computation is complete we can scale the gradients
                    # back to the correct scale before passing them to the optimizer.
                    unscaled_grad_vars = [(grad / loss_scale, var)
                                            for grad, var in scaled_grad_vars]
                    minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
                else:
                    minimize_op = optimizer.minimize(loss, global_step)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = tf.group(minimize_op, update_ops)
            else:
                train_op = None

            if not tf.contrib.distribute.has_distribution_strategy():
                accuracy = tf.metrics.accuracy(
                    tf.argmax(labels, axis=1), predictions["classes"])
            else:
                # Metrics are currently not compatible with distribution strategies during
                # training. This does not affect the overall performance of the model.
                accuracy = (tf.no_op(), tf.constant(0))

            metrics = {"accuracy": accuracy}

            # Create a tensor named train_accuracy for logging purposes
            tf.identity(accuracy[1], name="train_accuracy")
            tf.summary.scalar("train_accuracy", accuracy[1])

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)

        return func

    def input_fn(self, tfrecord_file, batch_size, is_training):
        # Create and return input function
        def func():
            # Initialize TFRecordDataset from file
            dataset = tf.data.TFRecordDataset([tfrecord_file])

            # We prefetch a batch at a time, This can help smooth out the time taken to
            # load input files as we go through shuffling and processing.
            dataset = dataset.prefetch(buffer_size=batch_size)
            if is_training:
                # Shuffle the records. Note that we shuffle before repeating to ensure
                # that the shuffling respects epoch boundaries.
                dataset = dataset.shuffle(buffer_size=1500)

            # If we are training over multiple epochs before evaluating, repeat the
            # dataset for the appropriate number of epochs.
            dataset = dataset.repeat(1)

            # Parse the raw records into images and labels. Testing has shown that setting
            # num_parallel_batches > 1 produces no improvement in throughput, since
            # batch_size is almost always much greater than the number of CPU cores.
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    lambda value: parse_record_fn(value, is_training),
                    batch_size=batch_size,
                    num_parallel_batches=1))

            # Operations between the final prefetch and the get_next call to the iterator
            # will happen synchronously during run time. We prefetch here again to
            # background all of the above processing work and keep it out of the
            # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
            # allows DistributionStrategies to adjust how many batches to fetch based
            # on how many devices are present.
            dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            
            return dataset
        return func

    def train(self, num_epochs, batch_size, checkpoint_dir=None):
        tf.logging.set_verbosity(tf.logging.INFO)

        train_input_fn = self.input_fn("tiny_imagenet_train.tfrecord", batch_size, True)
        val_input_fn = self.input_fn("tiny_imagenet_val.tfrecord", batch_size, False)

        train_data_size = get_dataset_size("tiny_imagenet_train.tfrecord")
        val_data_size = get_dataset_size("tiny_imagenet_val.tfrecord")

        learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
            batch_size=batch_size, batch_denom=256,
            num_images=train_data_size, boundary_epochs=[30, 60, 80, 90],
            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

        warm_start = None
        if checkpoint_dir != None:
            # Load pre-trained model for all variables except resnet_model/dense*
            warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=checkpoint_dir,
                                                        vars_to_warm_start="^((?!resnet_model\/dense).)*$")

        estimator = tf.estimator.Estimator(model_fn=self.model_fn(learning_rate_fn=learning_rate_fn, loss_filter_fn=None), model_dir=get_log_dir("models"), warm_start_from=warm_start, config=config)
        
        # Iterate epochs
        for epoch in range(num_epochs):
            tf.logging.info("*** Epoch {}/{} ***".format(epoch + 1, num_epochs))
            
            tf.logging.info("*** Training ***")
            estimator.train(input_fn=train_input_fn)
            
            tf.logging.info("*** Evaluating ***")
            estimator.evaluate(input_fn=val_input_fn)

if __name__ == "__main__":
    import argparse
    
    # Default training parameters
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_NUM_EPOCHS = 100

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the training procedure", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size",
                        help="Size of the batches",
                        type=int,
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_epochs",
                        help="Number of epochs to train",
                        type=int,
                        default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--checkpoint_dir",
                        help="Directory of previously trained model",
                        type=str,
                        default=None)
    args = parser.parse_args()

    # Train
    trainer = FasterRCNNTrainer()
    trainer.train(args.num_epochs, args.batch_size, args.checkpoint_dir)