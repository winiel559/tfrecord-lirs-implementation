import tensorflow as tf
import numpy as np
import time
import random
import os
import sys
import struct
import threading
from queue import Queue

CUDA_N = sys.argv[5]
print(CUDA_N, type(CUDA_N), "\n\n\n\n\n\n")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

# Define parameters
#DATA_DIR = "/home/xpoint/TensorFlow/ImageNet/imagenet_1280K/"
DATA_DIR = "/media/ssd/imagenet_1280K/"
NUM_INSTANCE = 1281024  # 1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = int(sys.argv[2])
MAX_EPOCH = 96
Q_MAXSIZE = int(sys.argv[3])
LEARNING_RATE = 1e-4
VAL_NUM = 50000
VAL_BATCH_SIZE = 200

if sys.argv[1] == 'fix_order':
    RANDOM_SHUFFLE = 0
    TF_QUEUE = 0
if sys.argv[1] == 'random_shuffle':
    RANDOM_SHUFFLE = 1
    TF_QUEUE = 0
if sys.argv[1] == 'tf_queue':
    RANDOM_SHUFFLE = 0
    TF_QUEUE = 1

# Initialize weight


def weight_variable(shape, stddev=0.001):
    initial = tf.truncated_normal(shape, mean=0, stddev=stddev)
    return tf.Variable(initial)

# Initialize bias


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Define convolutional layer


def conv2d(x, W, stride_y, stride_x, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride_y, stride_x, 1], padding=padding)

# Define max pooling layer with stride parameter


def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding)

# Local response normalization from AlexNet


def lrn(x, radius, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

# Add CONV layer or FC layer


def add_layer(inputs, shape, layer_type, stddev=0.001, act_func=None, stride_y=1, stride_x=1, padding='SAME', norm=True):
    Weights = weight_variable(shape, stddev)

    if layer_type == 'CONV':
        out_size = shape[3]
        biases = bias_variable([out_size])
        Wx_plus_b = conv2d(inputs, Weights, stride_y,
                           stride_x, padding) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0, 1, 2],)
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(
                Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

    if layer_type == 'FC':
        out_size = shape[1]
        biases = bias_variable([out_size])
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if norm:
            fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0],)
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            Wx_plus_b = tf.nn.batch_normalization(
                Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

    if act_func is None:
        outputs = Wx_plus_b
    else:
        outputs = act_func(Wx_plus_b)
    return outputs

# Get one image [256*256*3, 1] from binary file


def get_image(index, train_img_dir, train_label_dir, istrain=True):

    image_offset = 0 + index*(256*256*3)
    train_img_dir.seek(image_offset, 0)
    #img = struct.unpack('>196608B', train_img_dir.read(196608))
    #img = np.array(img).astype(np.float32)
    buf = train_img_dir.read(196608)
    img = np.fromstring(buf, dtype='>B').astype(np.uint8)
    # print(buf,img)
    # print(img.shape)
    img = np.reshape(img, [256, 256, 3])
    # print(img.shape)
    # data aug: random_crop, random_flip, PCA
    if istrain:
        x = random.randint(0, 28)
        y = random.randint(0, 28)
        img = img[y:y+227, x:x+227]

        r = random.randint(0, 1)
        if r == 1:
            img = np.fliplr(img)
        """
        renorm_image = np.reshape(img,(51529,3))

        renorm_image = renorm_image.astype('float32')
        renorm_image -= np.mean(renorm_image, axis=0)
        renorm_image /= np.std(renorm_image, axis=0)

        cov = np.cov(renorm_image, rowvar=False)

        lambdas, p = np.linalg.eig(cov)
        alphas = np.random.normal(0, 0.1, 3)

        delta = np.dot(p, alphas*lambdas)
        mean = np.mean(renorm_image, axis=0)
        std = np.std(renorm_image, axis=0)
        pca_augmentation_version_renorm_image = renorm_image + delta
        pca_color_image = pca_augmentation_version_renorm_image * std + mean
        pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0)
        img = np.reshape(pca_color_image,(227,227,3))
        """

    else:
        img = img[14:241, 14:241]
    label_offset = 0 + index*(2)
    train_label_dir.seek(label_offset, 0)

    #label = struct.unpack('>1H', train_label_dir.read(2))
    #label = np.array(label).astype(np.float32)
    buf = train_label_dir.read(2)
    label = np.fromstring(buf, dtype='>H').astype(np.float32)

    img = (img.astype(np.float32)/255)
    label = label-1
    # print(img.shape)
    return img, label

# Enqueue the training data into Queue q


def enqueue(index_list, q, train_img_dir, train_label_dir, istrain=True):

    for index in index_list:
        image, label = get_image(index, train_img_dir,
                                 train_label_dir, istrain)
        q.put([image, label])

    train_img_dir.close()
    train_label_dir.close()

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
    #example_batch = np.array(example_batch).astype(np.float32)
    #label_batch = np.array(label_batch).astype(np.int32)
    # endtime=time.time()
    #print("for loop:",endtime-midtime)
    #print("convert to nparray",midtime-starttime)
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

    #example_batch = np.array(example_batch).astype(np.float32)
    #label_batch = np.array(label_batch).astype(np.int32)

    return example_batch, label_batch

# Create threads to do the enqueue process


def Create_threads(training_order, train_img_list, train_label_list):
    threads = []
    for i in range(NUM_THREADS):
        train_img_dir = open(train_img_list[i], 'rb')
        train_label_dir = open(train_label_list[i], 'rb')
        t = threading.Thread(target=enqueue,
                             args=(training_order[i], q, train_img_dir, train_label_dir, True))
        t.start()
        threads.append(t)
    return threads


# Define model
with tf.name_scope('model'):
    x = tf.placeholder('float32', shape=[None, 227, 227, 3])
    y_raw = tf.placeholder('int32', shape=[None, 1])
    y_ = tf.cast(tf.one_hot(tf.reshape(y_raw, [-1]), depth=1000), tf.float32)
    keep_prob = tf.placeholder('float32')

    #x = tf.placeholder('float32', shape=[None, 256*256*3])
    #x_image_ba = tf.reshape(x, [-1, 256, 256, 3])
    
    # x_image_ba=x

    istrain = tf.placeholder('bool')
    lr = tf.placeholder('float32')
    # tf.cond(istrain, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(tf.random_crop(img, [227, 227, 3])),x_image_ba), lambda: tf.map_fn(lambda img: tf.image.central_crop(img, 0.8867),x_image_ba))
    x_image = x
    conv1 = add_layer(x_image, [11, 11, 3, 96], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=4, stride_x=4, padding='VALID')
    #norm1 = lrn(conv1, 2, 2e-05, 0.75)
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')

    conv2 = add_layer(pool1, [5, 5, 96, 256], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
    #norm2 = lrn(conv2, 2, 2e-05, 0.75)
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID')

    conv3 = add_layer(pool2, [3, 3, 256, 384], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')

    conv4 = add_layer(conv3, [3, 3, 384, 384], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')

    conv5 = add_layer(conv4, [3, 3, 384, 256], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

    flat = tf.reshape(pool5, [-1, 6*6*256])

    FC_1 = add_layer(flat, [6*6*256, 4096], 'FC',
                     stddev=0.1, act_func=tf.nn.relu)
    DROP_1 = tf.nn.dropout(FC_1, keep_prob)
    FC_2 = add_layer(DROP_1, [4096, 4096], 'FC',
                     stddev=0.1, act_func=tf.nn.relu)
    DROP_2 = tf.nn.dropout(FC_2, keep_prob)
    y_conv = add_layer(DROP_2, [4096, 1000], 'FC', stddev=0.1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

if __name__ == '__main__':

    # Create a queue with size Q_MAXSIZE to buffer the training data
    q = Queue(maxsize=Q_MAXSIZE)

    # Create the training order list
    training_order = np.zeros((NUM_THREADS, NUM_INSTANCE)).astype(np.int32)
    for i in range(NUM_THREADS):
        for j in range(NUM_INSTANCE):
            training_order[i][j] = j

    # Create the training file list
    train_img_list = []
    train_label_list = []

    for i in range(1):
        train_img_list.append(DATA_DIR + "train-images")
        train_label_list.append(DATA_DIR + "train-labels")

    # Reader threads list
    threads = []

    # Example list used in TF_queue
    example_list = []

    # Initialize timing parameter
    loadtime = 0
    traintime = 0
    startTime = time.time()

    # Initialize TensorFlowi
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)

    for k in range(NUM_THREADS):
        random.shuffle(training_order[k])

    # Start training
    for i in range(MAX_EPOCH):
        if i in [30, 50, 60, 70, 80]:
            LEARNING_RATE = LEARNING_RATE/2
            print("val time epoch ", i, " learing rate changes to ", LEARNING_RATE)
        # If the mode is RANDOM_SHUFFLE, shuffle the training order list first
        if RANDOM_SHUFFLE:
            for k in range(NUM_THREADS):
                random.shuffle(training_order[k])

        # Create threads(Readers) to read the training data
        threads = Create_threads(
            training_order, train_img_list, train_label_list)

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
            _, train_accuracy, train_loss = sess.run([train_step, accuracy, loss], feed_dict={
                                                     x: batch[0], y_raw: batch[1], keep_prob: 0.5, lr: LEARNING_RATE, istrain: True})
            # print(x_image.shape)
            # exit()
            traintime += (time.time()-start_train)

            print("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g" % (
                i, j, train_loss, train_accuracy))

            # Print the time every 100 iters
            if j % (128000/BATCH_SIZE) == 0 and (i != 0 or j != 0):
                # Create threads and enqueue validation data
                v_q = Queue(maxsize=Q_MAXSIZE)

                v_threads = []
                v_image = open(DATA_DIR+'val-images', 'rb')
                v_label = open(DATA_DIR+'val-labels', 'rb')
                v_order = np.zeros(VAL_NUM).astype(np.int32)
                for k in range(VAL_NUM):
                    v_order[k] = k

                t = threading.Thread(target=enqueue, args=(
                    v_order, v_q, v_image, v_label, False))
                t.start()
                v_threads.append(t)

                # Evaluate the validation data
                val_accuarcy = 0
                val_loss = 0
                for k in range(int(VAL_NUM/VAL_BATCH_SIZE)):
                    v_batch = next_batch(v_q, batch_size=VAL_BATCH_SIZE)
                    v_accuarcy, v_loss = sess.run([accuracy, loss], feed_dict={
                                                  x: v_batch[0], y_raw: v_batch[1], keep_prob: 1.0, istrain: False})
                    val_accuarcy = val_accuarcy + v_accuarcy
                    val_loss = val_loss + v_loss

                print("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g, val_loss, %6g, val_accuracy, %g"
                      % (i, j, train_loss, train_accuracy, val_loss/(VAL_NUM/VAL_BATCH_SIZE), val_accuarcy/(VAL_NUM/VAL_BATCH_SIZE)))

                for thread in v_threads:
                    thread.join()

        # Sync the threads
        for thread in threads:
            thread.join()

        print("Total time: %f Traintime: %f Loadtime: %f" %
              ((time.time() - startTime), traintime, loadtime))
        saver_path = saver.save(sess, './alexnet_model/am')
