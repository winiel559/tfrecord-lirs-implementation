import tensorflow as tf

import numpy as np
import time
import random
import os
import sys
import struct
import threading
from queue import Queue
# Define parameters
# DATA_DIR = "/home/xpoint/TensorFlow/ImageNet/imagenet_1280K/"
DATA_DIR = "./"
NUM_INSTANCE = 60000  # 1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = 200
MAX_EPOCH = 96
Q_MAXSIZE = 400
LEARNING_RATE = 1e-4

if sys.argv[1] == 'fix_order':
    RANDOM_SHUFFLE = 0
    TF_QUEUE = 0
if sys.argv[1] == 'random_shuffle':
    RANDOM_SHUFFLE = 1
    TF_QUEUE = 0
if sys.argv[1] == 'tf_queue':
    RANDOM_SHUFFLE = 0
    TF_QUEUE = 1

# Get one image [256*256*3, 1] from binary file


"""
def get_image(index, train_img_dir, train_label_dir, istrain=True):

    image_offset = 0 + index*(256*256*3)
    train_img_dir.seek(image_offset, 0)
    # img = struct.unpack('>196608B', train_img_dir.read(196608))
    # img = np.array(img).astype(np.float32)
    buf = train_img_dir.read(196608)
    img = np.fromstring(buf, dtype='>B').astype(np.uint8)
    # print(buf,img)
    # print(img.shape)
    img = np.reshape(img, [256, 256, 3])

    label_offset = 0 + index*(2)
    train_label_dir.seek(label_offset, 0)

    # label = struct.unpack('>1H', train_label_dir.read(2))
    # label = np.array(label).astype(np.float32)
    buf = train_label_dir.read(2)
    label = np.fromstring(buf, dtype='>H').astype(np.float32)

    img = (img.astype(np.float32)/255)
    label = label-1
    # print(img.shape)
    return img, label
"""

""" the actual read() function """
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse_function(example_proto):
    serialized_example = tf.io.parse_single_example(
        example_proto, image_feature_description)
    return serialized_example


def get_non_sparse_instance_length(binfile):
    tmp = binfile.read(8)
    length = struct.unpack("<Q", tmp)
    print(length)
    record_l = length[0]+16
    return record_l


def non_sparse_random_read(binfile, idx, record_l=-1):
    # calc one record length (B) if not specified
    if(record_l < 0):
        # read data length
        tmp = binfile.read(8)
        length = struct.unpack("<Q", tmp)
        # include the 0,2,3 column
        record_l = length[0]+16

    # random read
    binfile.seek(idx*record_l, 0)
    tmp = binfile.read(record_l)
    r_data = tmp[12:-4]

    # deserialize data
    parsed_features = _parse_function(r_data)
    # for key in parsed_features:
    #    print(f"{key}: {parsed_features[key].numpy()}")

    return parsed_features


""" end of the actual read() function """


def get_image(index, dataset, record_l=-1):
    parsed_features = non_sparse_random_read(dataset, index, record_l)
    img = parsed_features['image'].numpy()
    img = np.frombuffer(img, dtype='>B').astype(np.uint8)
    img = np.reshape(img, [28, 28, 1])
    img = (img.astype(np.float32)/255)
    label = parsed_features['label'].numpy()
    label = np.frombuffer(label, dtype='>B').astype(np.uint8)
    label = label.astype(np.float32)
    return img, label

# Enqueue the training data into Queue q


def enqueue(index_list, q, dataset, record_l=-1):
    for index in index_list:
        image, label = get_image(index, dataset, record_l)
        q.put([image, label])

    dataset.close()

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


def next_batch_TF(q, example_list, notTail, batch_size=BATCH_SIZE):
    example_batch = []
    label_batch = []

    while(len(example_list) < Q_MAXSIZE and notTail):
        example_list.append(q.get())

    # Random shuffle the queue then dequeue
    random.shuffle(example_list)

    for i in range(batch_size):
        example, label = example_list.pop()
        example_batch.append(example)
        label_batch.append(label)

    # example_batch = np.array(example_batch).astype(np.float32)
    # label_batch = np.array(label_batch).astype(np.int32)

    return example_batch, label_batch

# Create threads to do the enqueue process


def Create_threads(training_order, dataset_list):
    threads = []
    for i in range(NUM_THREADS):
        dataset = open(dataset_list[i], 'rb')
        record_l=get_non_sparse_instance_length(dataset)
        print("record_l=",record_l)
        t = threading.Thread(target=enqueue,
                             args=(training_order[i], q, dataset,record_l))
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

    # Create a queue with size Q_MAXSIZE to buffer the training data
    q = Queue(maxsize=Q_MAXSIZE)

    # Create the training order list
    training_order = np.zeros((NUM_THREADS, NUM_INSTANCE)).astype(np.int32)
    for i in range(NUM_THREADS):
        for j in range(NUM_INSTANCE):
            training_order[i][j] = j

    # Create the training file list
    dataset_list = []

    for i in range(1):
        dataset_list.append(DATA_DIR + "mnist_nonsparse.tfrecords")

    # Reader threads list
    threads = []

    # Example list used in TF_queue
    example_list = []

    # Initialize timing parameter
    loadtime = 0
    traintime = 0
    startTime = time.time()

    """
    # Initialize TensorFlow
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)
    """
    for k in range(NUM_THREADS):
        random.shuffle(training_order[k])

    # Start training
    for i in range(MAX_EPOCH):
        # If the mode is RANDOM_SHUFFLE, shuffle the training order list first
        if RANDOM_SHUFFLE:
            for k in range(NUM_THREADS):
                random.shuffle(training_order[k])

        # Create threads(Readers) to read the training data
        threads = Create_threads(
            training_order, dataset_list)

        train_loss.reset_states()
        train_accuracy.reset_states()
        #test_loss.reset_states()
        #test_accuracy.reset_states()
        # Inner iterations in range(Total data/batch size)
        for j in range(int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)):

            # Start getting batch from Queue q
            start_getbatch = time.time()

            # import data for TensorFlow input pipeline
            if TF_QUEUE:
                if(j < int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)-(Q_MAXSIZE/BATCH_SIZE-1)):
                    notTail = 1
                else:
                    notTail = 0
                batch = next_batch_TF(q, example_list, notTail, BATCH_SIZE)

            # import data for fix_order and random shuffle
            if not TF_QUEUE:
                batch = next_batch(q, BATCH_SIZE)

            
            loadtime += (time.time()-start_getbatch)

            # Start training process
            start_train = time.time()
            train_step(batch[0],batch[1])
            traintime += (time.time()-start_train)

        print("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g" % (
                i, j, train_loss.result(), train_accuracy.result()))

        # Sync the threads
        for thread in threads:
            thread.join()

        print("Total time: %f Traintime: %f Loadtime: %f" %
              ((time.time() - startTime), traintime, loadtime))
