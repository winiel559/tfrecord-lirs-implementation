# This version only supports padding with page_assign, not pure page_assign

import tensorflow as tf
import numpy as np
import time
import random
import os
import sys
import struct
import threading
from queue import Queue
from TFRecordLIRS import TFRecordLIRS

# Define parameters


PAGE_ASSIGN = sys.argv[1]
TFR_DIR = sys.argv[2]
OFT_DIR = sys.argv[3]

NUM_INSTANCE = 60000  # 1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = 200
MAX_EPOCH = 96
Q_MAXSIZE = 400
LEARNING_RATE = 1e-4
RANDOM_SHUFFLE = 1


def enqueue(train_dataset, NUM_INSTANCE, q):
    for _ in range(NUM_INSTANCE):
        image, label = train_dataset.next()
        q.put([image, label])
        #print(label, end=' ')

# Dequeue and form a batch size of data (Used in random_shuffle mode)


def next_batch(q, batch_size=BATCH_SIZE):
    example_batch = []
    label_batch = []
    # starttime=time.time()
    for i in range(batch_size):
        example, label = q.get()
        example_batch.append(example)
        label_batch.append(label)
    # midtime=time.time()
    example_batch = np.array(example_batch).astype(np.float32)
    label_batch = np.array(label_batch).astype(np.float32)
    # endtime=time.time()
    # print("for loop:",endtime-midtime)
    # print("convert to nparray",midtime-starttime)
    return example_batch, label_batch

# Dequeue and form a batch size of data (Used in TF_queue mode)
# It will random shuffle the q first.


def Create_threads(train_dataset, NUM_INSTANCE, q):
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=enqueue,
                             args=(train_dataset, NUM_INSTANCE, q))
        t.start()
        threads.append(t)
    return threads


# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
]
)
# Define loss_fn and optiminzer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

# Define training step


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


if __name__ == '__main__':

    # the description of tfr file
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }    

    train_dataset = TFRecordLIRS(image_feature_description, TFR_DIR, OFT_DIR, NUM_INSTANCE)

    # Create a queue with size Q_MAXSIZE to buffer the training data
    q = Queue(maxsize=Q_MAXSIZE)


    # Reader threads list
    threads = []

    # Example list used in TF_queue
    example_list = []

    # Initialize timing parameter
    loadtime = 0
    traintime = 0
    startTime = time.time()

    train_dataset.shuffle_order(PAGE_ASSIGN)

    # Start training
    for i in range(MAX_EPOCH):
        # If the mode is RANDOM_SHUFFLE, shuffle the training order list first
        if RANDOM_SHUFFLE:
            train_dataset.shuffle_order(PAGE_ASSIGN)

        # Create threads(Readers) to read the training data
        threads = Create_threads(
            train_dataset, NUM_INSTANCE, q)

        train_loss.reset_states()
        train_accuracy.reset_states()
        # Inner iterations in range(Total data/batch size)
        for j in range(int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)):

            # Start getting batch from Queue q
            start_getbatch = time.time()

            # import data for fix_order and random shuffle
            #if not TF_QUEUE:
            batch = next_batch(q, BATCH_SIZE)

            loadtime += (time.time()-start_getbatch)

            # Start training process
            start_train = time.time()
            train_step(batch[0], batch[1])
            traintime += (time.time()-start_train)

        print("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g" % (
            i, j, train_loss.result(), train_accuracy.result()))

        # Sync the threads
        for thread in threads:
            thread.join()

        print("Total time: %f Traintime: %f Loadtime: %f" %
              ((time.time() - startTime), traintime, loadtime))
