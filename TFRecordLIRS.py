import tensorflow as tf
import numpy as np
import random
import struct
import threading


class TFRecordLIRS:
    def __init__(self, feature_description, tfr_filename, oft, NUM_INSTANCE):
        self.feature_description = feature_description
        # self.tfr_filename=tfr_filename
        self.binfile = open(tfr_filename, 'rb')
        self.oft = open(oft, 'rb')
        self.NUM_INSTANCE = NUM_INSTANCE
        self.NUM_THREADS = 1
        self.index = 0
        # Create the training order list
        self.training_order = np.zeros((1, self.NUM_INSTANCE)).astype(np.int32)
        for i in range(self.NUM_THREADS):
            for j in range(self.NUM_INSTANCE):
                self.training_order[i][j] = j

    def _parse_function(self, example_proto):
        serialized_example = tf.io.parse_single_example(
            example_proto, self.feature_description)
        return serialized_example

    def shuffle_order(self, page_assign=False):
        for k in range(self.NUM_THREADS):
            random.shuffle(self.training_order[k])

    def sparse_random_read(self, idx):
        # read offset, 8B per instance
        # TODO: oft should be loaded to mem
        self.oft.seek(idx*8, 0)
        tmp = self.oft.read(8)
        offset = struct.unpack("<Q", tmp)[0]
        # print("offset=",offset)

        # read data length
        self.binfile.seek(offset, 0)
        tmp = self.binfile.read(8)
        length = struct.unpack("<Q", tmp)[0]
        # print("length=",length)
        # we already read first 8B
        record_l_from_col2 = length+8

        # random read
        tmp = self.binfile.read(record_l_from_col2)
        r_data = tmp[4:-4]

        # deserialize data
        parsed_features = self._parse_function(r_data)
        return parsed_features

    def _next(self, instance_idx):
        parsed_features = self.sparse_random_read(instance_idx)
        # decode jpeg string to pixel array
        img = tf.io.decode_jpeg(parsed_features['image'])
        img = np.frombuffer(img, dtype='>B').astype(np.uint8)
        img = np.reshape(img, [28, 28, 1])
        img = (img.astype(np.float32)/255)
        label = parsed_features['label'].numpy()
        label = np.frombuffer(label, dtype='>B').astype(np.uint8)
        label = label.astype(np.float32)
        return img, label

    def next(self):
        instance_idx = self.training_order[0][self.index]
        self.index += 1
        if(self.index >= self.NUM_INSTANCE):
            self.index = 0
        return self._next(instance_idx)
