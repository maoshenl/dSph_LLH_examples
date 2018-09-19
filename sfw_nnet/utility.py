import os
import numpy as np
import tensorflow as tf
import re


# writing the kernel density estimate 30x30 grid-point values representation of each galaxy
# to TFrecord files.
# Parameter: 
#     tffname: the tfrecod file name
#     setdir: the directory where the training (or validation) data is located; saved as .npy
#        -- each file contains 'bsize' rows by 911 columns; 2D array
#        -- each column represent a galaxy, the first 900 elements are the KDE values;
#           the last 11 values are the corresponding parameters, normalized to within [0,1]
def write_30kde_p11_npy(tffname, setdir):
    filelist = os.listdir(setdir) 

    writer = tf.python_io.TFRecordWriter(tffname)
    print 'Writing', tffname

    for i, fi in enumerate(filelist): #looping over files in the directory
        print i, ' / ', len(filelist)
        filename = setdir + fi

        batch_XY = np.load(filename) 
        batch_XY = np.round(batch_XY, 8)
        bsize = np.shape(batch_XY)[0]
        for j in xrange(bsize): #looping over the galaxies in a given file
            XY = batch_XY[j,:]
            X = XY[0:900] #KDE 30x30 grid density values
            Y = XY[900:]  #parameters

            # write to tfrecord file
            datasize = np.size(X)
            data_raw = X.tostring() #
            example = tf.train.Example(features=tf.train.Features(feature={
                'data_size': _int64_feature(datasize),
                'label': _float_feature(Y),
                'data_raw': _bytes_feature(data_raw)}))
            writer.write(example.SerializeToString())

    writer.close()
    return 0 


# reading the tfrecod written using 'write_30kde_p11_npy' above.
# Parameters:
#      datasize: size of training data; should be 900 as in function 'write_30kde_p11_npy'
#      bsize: training batch size
#      capacity: capacity is the maximum size of queue 
#      min_after_dequeue: minimum size of queue after dequeue
#      label_size: size of the label array; should be 11 as in function 'write_30kde_p11_npy'
# capacity is the maximum size of queue, min_after_dequeue is the minimum size of queue after dequeue,
def read_data(filenames, datasize, bsize=5, capacity=100, min_after_dequeue=10, label_size=11):

        reader = tf.TFRecordReader()

        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        _, serialized_example = reader.read(filename_queue)

        feature={
                'data_size':  tf.FixedLenFeature([], tf.int64),
                'label':  tf.FixedLenFeature([label_size],   dtype=tf.float32),
                'data_raw': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
        features = tf.parse_single_example(serialized_example, features=feature)

        X = tf.decode_raw(features['data_raw'], tf.float64)

        X = tf.reshape(X, [datasize,1])
        label = tf.cast(features['label'], tf.float64)
        #label = tf.reshape(label, (3,))

        Xs, labels = tf.train.shuffle_batch( [X, label],
                                                 batch_size= bsize,
                                                 capacity=capacity,
                                                 num_threads=2,
                                                 min_after_dequeue=min_after_dequeue)
        return Xs, labels


#----------------------------------------------------------------------
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


