import tensorflow as tf
import time
import numpy as np
import pickle

from transformer.model import Transformer
from transformer.model import create_padding_mask
from transformer.model import create_look_ahead_mask
from transformer.optimizer import CustomSchedule

feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    sequence = parsed_example['sequence'].values
    return mz, sequence

path_train_data='./data/preprocessed_train_data.tfrecords'
path_valid_data='./data/preprocessed_valid_data.tfrecords'
path_test_data='./data/preprocessed_test_data.tfrecords'

train_dataset = tf.data.TFRecordDataset(path_train_data)
valid_dataset = tf.data.TFRecordDataset(path_valid_data).map(parse_function)
test_dataset = tf.data.TFRecordDataset(path_test_data).map(parse_function)

#Load vocabulary
with open('./data/mz_vocab.pickle','rb') as f:
    mz_vocab = pickle.load(f)
with open('./data/AA_vocab.pickle','rb') as f:
    AA_vocab = pickle.load(f)

#Get vocabulary size
mz_vocab_size = len(mz_vocab)
AA_vocab_size = len(AA_vocab)

#Set batchs
'''
BATCH_SIZE = 64
train_batches = (train_dataset
                 .map(parse_function)
                 .padded_batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))
'''


def evaluate(dataset, max_length=50):
    for mz, sequence in dataset.padded_batch(1).take(1):
        print(mz, sequence)
        '''
        len_seq = sequence.shape[0]
        start, end = sequence[0], sequence[len_seq-1]
        print(start, end)

        tens = tf.convert_to_tensor([1])
        print(tf.expand_dims(tens,0))
        '''


emb = tf.convert_to_tensor(
    [
    [[1, 2, 3,5],
    [3, 3, 5,7],
    [4, 6, 7,8]],

    [[1, 2, 3,2],
    [3, 3, 5,1],
    [4, 6, 7,2]]
    ],
    dtype= tf.int64)

d_model = 4

inten = tf.convert_to_tensor(
        [[1, 2, 3],
         [3, 3, 3]],
    dtype=tf.int64)

print(inten[:,:,tf.newaxis])
inten = inten[:,:,tf.newaxis]
t = tf.repeat(inten, d_model, axis = 2)

print(t)

print(emb*t)