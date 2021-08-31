import tensorflow as tf
import numpy as np
import time
import random
import sys
import os
import struct
import threading
import cv2
import math
from queue import Queue
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define parameters
# DATA_DIR = "/home/xpoint/TensorFlow/ImageNet/imagenet_1280K/"   #Disk
DATA_DIR = "/media/rssd2/train/"  # SSD
NUM_INSTANCE = 1281024  # 1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = int(sys.argv[2])
MAX_EPOCH = 30
Q_MAXSIZE = int(sys.argv[3])
LEARNING_RATE = 1e-4
VAL_NUM = 50000
VAL_BATCH_SIZE = 200
MODEL_PATH = sys.argv[4]

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


def get_vimage(index, val_img_dir, val_label_dir):
    image_offset = 0 + index*(256*256*3)
    val_img_dir.seek(image_offset, 0)
    #img = struct.unpack('>196608B', train_img_dir.read(196608))
    #img = np.array(img).astype(np.float32)
    buf = val_img_dir.read(196608)
    img = np.fromstring(buf, dtype='>B').astype(np.float32)

    label_offset = 0 + index*(2)
    val_label_dir.seek(label_offset, 0)
    #label = struct.unpack('>1H', train_label_dir.read(2))
    #label = np.array(label).astype(np.float32)
    buf = val_label_dir.read(2)
    label = np.fromstring(buf, dtype='>H').astype(np.float32)

    img = (img/255)

    label = label-1
    return img, label

# Enqueue the training data into Queue q


def venqueue(index_list, q, val_img_dir, val_label_dir):
    for index in index_list:
        image, label = get_vimage(index, val_img_dir, val_label_dir)
        q.put([image, label])

    val_img_dir.close()
    val_label_dir.close()


def get_image(i, train_img_dir, train_label_dir):
    """
    image_offset = 0 + index*(256*256*3)
    train_img_dir.seek(image_offset, 0)
    #img = struct.unpack('>196608B', train_img_dir.read(196608))
    #img = np.array(img).astype(np.float32)
    buf = train_img_dir.read(196608)
    img = np.fromstring(buf, dtype='>B').astype(np.float32)
    """

    name = train_img_dir[i]
    img_path = DATA_DIR+'unzip/'+name
    img = cv2.imread(img_path)

    label_offset = 0 + int(name[:-4])*(2)
    train_label_dir.seek(label_offset, 0)
    #label = struct.unpack('>1H', train_label_dir.read(2))
    #label = np.array(label).astype(np.float32)
    buf = train_label_dir.read(2)
    label = np.fromstring(buf, dtype='>H').astype(np.float32)
    label = label-1
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype
    """
    if img.shape[0]<256 or img.shape[1]<256:
        #print("before",img.shape)
        if img.shape[0]<256:
            y=img.shape[0]>>1
            if img.shape[0]%2==0:
                
                img = cv2.copyMakeBorder(img,128-y,128-y,0,0,cv2.BORDER_CONSTANT)
            else:
                #y=math.floor(img.shape[0]/2)
                img = cv2.copyMakeBorder(img,128-y,127-y,0,0,cv2.BORDER_CONSTANT)
        if img.shape[1]<256:
            x=img.shape[1]>>1
            if img.shape[1]%2==0:
                #x=img.shape[1]/2
                img = cv2.copyMakeBorder(img,0,0,128-x,128-x,cv2.BORDER_CONSTANT)
            else:
                #x=math.floor(img.shape[0]/2)
                img = cv2.copyMakeBorder(img,0,0,128-x,127-x,cv2.BORDER_CONSTANT)
    """
    # print(img.shape)
    #img = np.asarray(img,dtype='float32')
    img = img.astype(np.float32)/255.0

    # print(img.shape)
    img = img.flatten()
    #print("flatten: ",img.shape)
    #raw_label = name[0:9]
    #label = np.array([label_dict[raw_label]], dtype='int32')
    #print(name, label)
    #label = label_dict[raw_label]

    return img, label

# Enqueue the training data into Queue q


def enqueue(q, train_img_dir, train_label_dir):
    c = 0
    for i in range(NUM_INSTANCE):
        image, label = get_image(i, train_img_dir, train_label_dir)
        if image.shape[0] == 196608:
            q.put([image, label])
        else:
            print("oh", image.shape[0])
            c += 1
    print("empth queue, wrong pic number:", c, "\n\n\n")
    # train_img_dir.close()
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
    # print(len(example_batch),example_batch[0].shape)
    # print(len(label_batch),label_batch[0])
    example_batch = np.array(example_batch).astype(np.float32)
    label_batch = np.array(label_batch).astype(np.int32)
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

    return example_batch, label_batch

# Create threads to do the enqueue process


def Create_threads(train_img_list, label_dict):  # , train_label_list):
    threads = []
    for i in range(NUM_THREADS):
        #train_img_dir = open(train_img_list[i], 'rb')
        train_label_dir = open("/media/rssd2/train-labels", 'rb')
        t = threading.Thread(target=enqueue,
                             args=(q, train_img_list, train_label_dir))  # ,train_label_dir))

        t.start()
        threads.append(t)
    return threads


# Define model
with tf.name_scope('model'):

    x = tf.placeholder('float32', shape=[None, 256*256*3])
    y_raw = tf.placeholder('int32', shape=[None, 1])
    y_ = tf.cast(tf.one_hot(tf.reshape(y_raw, [-1]), depth=1000), tf.float32)
    keep_prob = tf.placeholder('float32')
    #x_image = tf.reshape(x, [-1, 227, 227, 3])
    #x_image =  tf.placeholder('float32', shape=[None, 256, 256, 3])
    x_image_ba = tf.reshape(x, [-1, 256, 256, 3])
    istrain = tf.placeholder('bool')
    x_image = tf.cond(istrain, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(tf.random_crop(
        img, [227, 227, 3])), x_image_ba), lambda: tf.map_fn(lambda img: tf.image.central_crop(img, 0.8867), x_image_ba))

    conv1 = add_layer(x_image, [11, 11, 3, 96], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=4, stride_x=4, padding='VALID')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')

    conv2 = add_layer(pool1, [5, 5, 96, 256], 'CONV', stddev=0.001,
                      act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
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
    # TODO:另一個版本沒用DROPOUT
    FC_2 = add_layer(DROP_1, [4096, 4096], 'FC',
                     stddev=0.1, act_func=tf.nn.relu)
    DROP_2 = tf.nn.dropout(FC_2, keep_prob)
    y_conv = add_layer(DROP_2, [4096, 1000], 'FC', stddev=0.1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

if __name__ == '__main__':

    # Create a queue with size Q_MAXSIZE to buffer the training data
    q = Queue(maxsize=Q_MAXSIZE)

    # Create the training order list
    #training_order = np.zeros((NUM_THREADS, NUM_INSTANCE)).astype(np.int32)
    # for i in range(NUM_THREADS):
    #    for j in range(NUM_INSTANCE):
    #        training_order[i][j] = j

    # Create the training file list

    train_img_list = os.listdir(DATA_DIR+'unzip/')
    # print(len(train_img_list))

    #name2id = open(DATA_DIR+"map_clsloc.txt", 'r')
    label_dict = {}
    """
    for line in name2id.readlines():
        name = line.split(' ')[0]
        idx = line.split(' ')[1]
        label_dict[name] = int(idx)-1
    """
    #print("dict formed")
    #print("img_list mem occupy: ", sys.getsizeof(train_img_list), " bytes")
    # for i in range(1):
    #train_img_list.append(DATA_DIR + "train-images")
    #train_label_list.append(DATA_DIR + "train-labels")

    # Reader threads list
    threads = []

    # Example list used in TF_queue
    example_list = []

    # Initialize timing parameter
    loadtime = 0
    traintime = 0
    startTime = time.time()

    # Initialize TensorFlow
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)

    # for k in range(NUM_THREADS):
    #    random.shuffle(training_order[k])

    # Start training
    for i in range(MAX_EPOCH):

        # If the mode is RANDOM_SHUFFLE, shuffle the training order list first
        # if RANDOM_SHUFFLE:
        #    for k in range(NUM_THREADS):
        #        random.shuffle(training_order[k])

        # Create threads(Readers) to read the training data
        # , train_label_list)
        random.shuffle(train_img_list)
        print("epoch ", i, " first ten img:")
        for j in range(10):
            print(train_img_list[j])
        #print("string list formed")

        threads = Create_threads(train_img_list, label_dict)
        #print("thread created")
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
                                                     x: batch[0], y_raw: batch[1], keep_prob: 0.5})
            #print (pool5_p)
            traintime += (time.time()-start_train)
            # img=img*255
            # for i in range(32):
            #    cv2.imwrite("img"+str(i)+".jpg",img[i])
            # exit()
            print("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g" % (
                i, j, train_loss, train_accuracy))
            #print ("Cross Entropy", cross_entropy_p)
            # Print the time every 100 iters
            if j % (128000/BATCH_SIZE) == 0 and (i != 0 or j != 0):
                # Create threads and enqueue validation data
                v_q = Queue(maxsize=Q_MAXSIZE)

                v_threads = []
                #v_image = open(DATA_DIR+'val-images', 'rb')
                #v_label = open(DATA_DIR+'val-labels', 'rb')
                v_image = open('/media/rssd2/val-images', 'rb')
                v_label = open('/media/rssd2/val-labels', 'rb')
                v_order = np.zeros(VAL_NUM).astype(np.int32)
                for k in range(VAL_NUM):
                    v_order[k] = k

                t = threading.Thread(target=venqueue, args=(
                    v_order, v_q, v_image, v_label))
                t.start()
                v_threads.append(t)

                # Evaluate the validation data
                val_accuarcy = 0
                val_loss = 0
                for k in range(int(VAL_NUM/VAL_BATCH_SIZE)):
                    v_batch = next_batch(v_q, batch_size=VAL_BATCH_SIZE)
                    v_accuarcy, v_loss = sess.run([accuracy, loss], feed_dict={
                                                  x: v_batch[0], y_raw: v_batch[1], keep_prob: 1.0})
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
        save_path = saver.save(sess, MODEL_PATH)
