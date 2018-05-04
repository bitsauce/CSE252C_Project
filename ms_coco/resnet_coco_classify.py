import os
import numpy as np
import tensorflow as tf
from model import MSCocoModel
from dataset import load_dataset
from pycocotools.coco import COCO

# Training parameters
BATCH_SIZE = 1
EPOCHS = 100

# Load MSCoco dataset
dataset = load_dataset(BATCH_SIZE)
itr = dataset.make_initializable_iterator()

# Initialize network
model = MSCocoModel(50)

# Feed image and labels through network
x, y = itr.get_next()
tf.summary.image("images", x, max_outputs=6)
logits = model(x, True)

# Calculate loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Create optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def get_log_dir(model_dir):
    run_idx = 0
    while True:
        log_dir = os.path.join(model_dir, "run_{}".format(run_idx))
        if not os.path.isdir(log_dir):
            return log_dir
        run_idx += 1

log_dir = get_log_dir("logs")

# Session
sess = tf.Session()

# 
# merged = tf.summary.merge_all()

# Create file writers
train_summary = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)

# Initialize models vars
sess.run(tf.global_variables_initializer())

def add_scalar_summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value),])

# Train loop
for epoch in range(EPOCHS):
    print("Epoch {}/{}: Training...".format(epoch + 1, EPOCHS))

    # Feed train data
    sess.run(itr.initializer)

    # Do one epoch
    n_iter = 0
    total_loss = 0.0
    total_accuracy = 0.0
    while True:
        try:
            _, loss_value, acc_value = sess.run([train_op, loss, accuracy])
            total_loss += loss_value
            total_accuracy += acc_value
            n_iter += 1
        except tf.errors.OutOfRangeError as e:
            break
        except Exception:
            raise

    # Save summary
    train_summary.add_summary(add_scalar_summary("average_loss", total_loss / n_iter), epoch)
    train_summary.add_summary(add_scalar_summary("average_accuracy", total_accuracy / n_iter), epoch)

    #writer.add_summary(summary)
    #summary = sess.run(merged, feed_dict={ average_loss_ph: total_loss / n_iter, accuracy_ph: total_accuracy / n_iter })

print("Train done")

train_summary.close()