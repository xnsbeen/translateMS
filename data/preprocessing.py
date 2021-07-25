import tensorflow as tf
import numpy as np

from data.msms import DataLoader
from data.msms import make_vocab
from data.msms import encode_to_int
from data.msms import save_preprocessed_data

data_loader = DataLoader()
dataset = data_loader.load('theoretical_raw_msms_data.tfrecords')

#record_size = None -> load whole data
record_size = 1000000

train_dataset, valid_dataset, test_dataset \
    = data_loader.split(dataset, record_size=record_size, train_prop=0.99, valid_prop=0.005)

max_num_peeks = 500

def prepare_data(dataset):
    global  max_num_peeks

    mz_list = []
    sequence_list = []
    intensity_list = []

    for record in dataset:
        #score = record['score'].numpy()
        mz = np.array(record['mz'].values)
        intensity = np.array(record['intensity'].values)
        sequence = record['sequence'].numpy().decode('utf-8')

        # Pick the index list of 500 largest intensity peeks
        index_max_intensity = np.sort((-intensity).argsort()[:max_num_peeks])

        mz = mz[index_max_intensity]
        intensity = intensity[index_max_intensity]

        mz_list.append(mz)
        sequence_list.append(list(sequence))

    return mz_list, sequence_list

train_mz_list, train_sequence_list = prepare_data(train_dataset)
valid_mz_list, valid_sequence_list = prepare_data(valid_dataset)
test_mz_list, test_sequence_list = prepare_data(test_dataset)

print(valid_sequence_list[:3])

#make vocablury about train dataset
mz_vocab = make_vocab(train_mz_list,"mz_vocab")
AA_vocab = make_vocab(train_sequence_list,"AA_vocab")

#encode and save train data
train_encoded_mz = encode_to_int(train_mz_list, mz_vocab)
train_encoded_sequence = encode_to_int(train_sequence_list, AA_vocab)
save_preprocessed_data(train_encoded_mz, train_encoded_sequence, 'preprocessed_train_data')

#encode and save validation data
valid_encoded_mz = encode_to_int(valid_mz_list, mz_vocab)
valid_encoded_sequence = encode_to_int(valid_sequence_list, AA_vocab)
save_preprocessed_data(valid_encoded_mz, valid_encoded_sequence, 'preprocessed_valid_data')

print(valid_encoded_sequence[:3])

#encode and save test data
test_encoded_mz = encode_to_int(test_mz_list, mz_vocab)
test_encoded_sequence = encode_to_int(test_sequence_list, AA_vocab)
save_preprocessed_data(test_encoded_mz, test_encoded_sequence, 'preprocessed_test_data')