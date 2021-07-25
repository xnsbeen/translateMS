import tensorflow as tf
import numpy as np
import pickle

class DataLoader:
    def __init__(self):
        self.feature_description = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            #'score': tf.io.FixedLenFeature([], tf.float32),
            'mz': tf.io.VarLenFeature(tf.int64),
            'intensity': tf.io.VarLenFeature(tf.float32),
        }

    def load(self,path_data='./data/raw_msms_data.tfrecords'):
        raw_dataset = tf.data.TFRecordDataset(path_data)
        parsed_dataset = raw_dataset.map(self.parse_function)
        return parsed_dataset

    def parse_function(self,example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, self.feature_description)

    def split(self, dataset, record_size=None, train_prop=0.99, valid_prop= 0.01,test_prop=0.01):
        #
        #dataset = dataset.shuffle(buffer_size=1000000)

        if record_size != None:
            dataset = dataset.take(record_size)

        train_size = int(record_size*train_prop)
        valid_size = int(record_size*valid_prop)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        valid_dataset = test_dataset.take(valid_size)
        test_dataset = test_dataset.skip(valid_size)

        return train_dataset, valid_dataset, test_dataset


def make_vocab(data,name):
    vocab = {'SOS' : 1, 'EOS':2}
    size_vocab = 2;

    for record in data:
        for value in record:
            if value not in vocab:
                size_vocab+=1
                vocab[value] = size_vocab

    path_vocab = '{}.pickle'.format(name)

    with open(path_vocab,'wb') as fw:
        pickle.dump(vocab, fw)

    return vocab

def encode_to_int(data,vocab):

    for i in range(len(data)):
        encoded_record = list()
        for j in range(0,len(data[i])):
            if data[i][j] in vocab:
                encoded_record.append(vocab[data[i][j]])
        data[i] = [vocab['SOS']] + encoded_record + [vocab['EOS']]

    return data


def save_preprocessed_data(mz_list, sequence_list, name):

    path = '{}.tfrecords'.format(name)

    def _int64_feature_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    with tf.io.TFRecordWriter(path) as writer:
        for mz, sequence in zip(mz_list, sequence_list):
            feature = {
                'mz': _int64_feature_list(mz),
                'sequence': _int64_feature_list(sequence),
            }
            serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(serialized_example.SerializeToString())