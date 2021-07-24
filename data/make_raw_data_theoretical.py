import numpy as np
import tensorflow as tf
import time

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    print(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

file_path = "C:/Users/ahoho/Downloads/Traning_TS_swiss_Carbam.txt"
f = open(file_path, "r")

record_file = 'theoretical_raw_msms_data.tfrecords'

starting_time = time.time()

with tf.io.TFRecordWriter(record_file) as writer:
    for i in range(int(1e7)) :
            if(i%10000==0) :
                print(f'[count] : {i}', f'[time] : {time.time()-starting_time}')

            record = f.readline()
            if not record : break

            record = record.split('\t')

            #bytes(utf-8)형으로 변환
            mz = np.array(100 * record[1:-2:2], dtype=np.float).astype(int)
            intensity = np.array(record[2:-2:2], dtype=np.float)
            sequence = record[-2].encode()

            feature = {
                'mz': _int64_feature_list(mz),
                'intensity': _float_feature_list(intensity),
                'sequence': _bytes_feature(sequence),
            }
            serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(serialized_example.SerializeToString())
