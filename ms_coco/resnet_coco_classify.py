import os
import numpy as np
import tensorflow as tf
from model import ResNetClassifier
from dataset import load_dataset

def get_log_dir(model_dir):
    run_idx = 0
    while True:
        log_dir = os.path.join(model_dir, "run_{}".format(run_idx))
        if not os.path.isdir(log_dir):
            return log_dir
        run_idx += 1

def add_scalar_summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value),])

# Training parameters
BATCH_SIZE = 2
EPOCHS = 10

# Load MSCoco dataset
dataset = load_dataset(BATCH_SIZE)
dataset = dataset.shuffle(buffer_size=1)
itr = dataset.make_initializable_iterator()

# Initialize network
model = ResNetClassifier(50)

# Feed image and labels through network
x, y = itr.get_next()
logits = model(x, True)

# Calculate loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Create optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create session
sess = tf.Session()

# Create file writers
train_summary = tf.summary.FileWriter(os.path.join(get_log_dir("logs"), "train"), sess.graph, flush_secs=1)

# Save example image
tf.summary.image("images", x, max_outputs=6)
merged = tf.summary.merge_all()

# Initialize models vars
sess.run(tf.global_variables_initializer())

# Train loop
for epoch in range(EPOCHS):
    print("Epoch {}/{}: Training...".format(epoch + 1, EPOCHS))

    # Feed train data
    sess.run(itr.initializer)

    # Do one epoch
    n_iter = 0
    total_loss = 0.0
    total_accuracy = 0.0
    summary = None
    while True:
        try:
            _, loss_value, acc_value, summary = sess.run([train_op, loss, accuracy, merged])
            total_loss += loss_value
            total_accuracy += acc_value
            n_iter += 1
        except tf.errors.OutOfRangeError as e:
            break
        except Exception:
            raise

    train_summary.add_summary(summary, epoch)

    # Save summary
    train_summary.add_summary(add_scalar_summary("average_loss", total_loss / n_iter), epoch)
    train_summary.add_summary(add_scalar_summary("average_accuracy", total_accuracy / n_iter), epoch)

print("Train done")

train_summary.close()