import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pyteomics import mgf

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
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

path_mgf = 'C:/Users/ahoho/Downloads/proteometools'
path_mgf_info = 'C:/Users/ahoho/Downloads/proteometools/info'

#title로 비교해서 m/z, intensity 가져오기 위해 dictionary 생성
dict_msms = dict()

for file_name in os.listdir(path_mgf_info):
    msms = pd.read_csv(path_mgf_info + '/' + file_name + '/msms.txt', delimiter='\t',low_memory=False)
    print(file_name)
    for idx, row in msms.iterrows():
        if row['Modifications'] == 'Unmodified':
            title = row['Raw file'] + '.' + str(row['Scan number']) + '.' + str(row['Scan number']) + '.' + str(
            row['Charge'])
            #필요한 정보 추가 가능
            score = row['Score']
            sequence = row['Sequence']
            dict_msms[title] = {'score': score, 'sequence': sequence}

record_file_train = 'data/real_raw_msms_data_train.tfrecords'
record_file_valid = 'data/real_raw_msms_data_valid.tfrecords'
record_file_test = 'data/real_raw_msms_data_test.tfrecords'

train_dict = dict()
valid_dict = dict()
test_dict = dict()

cnt = 0
with tf.io.TFRecordWriter(record_file_train) as writer_train, \
        tf.io.TFRecordWriter(record_file_valid) as writer_valid, \
        tf.io.TFRecordWriter(record_file_test) as writer_test:

    for file_name in os.listdir(path_mgf):
        print(file_name)
        if(file_name == 'info') : continue

        reader = mgf.read(path_mgf + '/' + file_name)

        for spectrum in reader:
            cnt+=1
            if(cnt%10000 == 0):
                print(cnt)
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

                if train_dict.get(sequence) == None and valid_dict.get(sequence) == None and test_dict.get(sequence) == None:
                    num = np.random.randint(10)
                    if(num == 1):
                        valid_dict[sequence] = 1
                    elif(num==2):
                        test_dict[sequence] = 1
                    else:
                        train_dict[sequence] = 1

                if sequence in train_dict:
                    writer_train.write(serialized_example.SerializeToString())
                elif sequence in valid_dict:
                    writer_valid.write(serialized_example.SerializeToString())
                elif sequence in test_dict:
                    writer_test.write(serialized_example.SerializeToString())

print('number of data :', cnt)
print('number of peptide in train dataset :', len(train_dict))
print('number of peptide in valid dataset :', len(valid_dict))
print('number of peptide in test dataset :', len(test_dict))
print('number of total peptide :', len(train_dict)+len(valid_dict)+len(test_dict))

'''
number of data : 6503421
number of peptide in train dataset : 85058
number of peptide in valid dataset : 10660
number of peptide in test dataset : 10684
number of total peptide : 106402
'''