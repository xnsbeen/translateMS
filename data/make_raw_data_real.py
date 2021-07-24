import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pyteomics import mgf, auxiliary

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

path_mgf = 'C:/Users/ahoho/Desktop/graduation_project/GA_mgf'
path_mgf_info = 'C:/Users/ahoho/Desktop/graduation_project/GA_mgf_info'

#title로 비교해서 m/z, intensity 가져오기 위해 dictionary 생성
dict_msms = dict()

for file_name in os.listdir(path_mgf_info):
    msms = pd.read_csv(path_mgf_info + '/' + file_name + '/msms.txt', delimiter='\t')
    print(file_name)
    for idx, row in msms.iterrows():
        title = row['Raw file'] + '.' + str(row['Scan number']) + '.' + str(row['Scan number']) + '.' + str(
            row['Charge'])
        #필요한 정보 추가 가능
        score = row['Score']
        sequence = row['Sequence']
        dict_msms[title] = {'score': score, 'sequence': sequence}

record_file = 'real_raw_msms_data.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:
    for file_name in os.listdir(path_mgf):
        reader = mgf.read(path_mgf + '/' + file_name)
        print(file_name)
        for spectrum in reader:
            #title = rawfile ~ ~ 중 맨 앞 rawfile만 가져옴
            params_title = spectrum['params']['title'].split()[0]
            mz = np.array(100 * spectrum['m/z array'], dtype=np.int)
            intensity = np.array(spectrum['intensity array'], dtype=np.float)

            if params_title in dict_msms:
                score = dict_msms[params_title]['score']
                #bytes(utf-8)형으로 변환
                sequence = dict_msms[params_title]['sequence'].encode()
                feature = {
                    'sequence': _bytes_feature(sequence),
                    'score': _float_feature(score),
                    'mz': _int64_feature_list(mz),
                    'intensity': _float_feature_list(intensity),
                }
                serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))

                writer.write(serialized_example.SerializeToString())
