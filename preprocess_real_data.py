import time
import numpy as np
import tensorflow as tf
import pickle

from data.msms import DataLoader

train_dataset_path = 'data/real_raw_msms_data_train.tfrecords'
valid_dataset_path = 'data/real_raw_msms_data_valid.tfrecords'
test_dataset_path = 'data/real_raw_msms_data_test.tfrecords'

preprocessed_train_data_path = 'data/real_preprocessed_train_data'
preprocessed_valid_data_path = 'data/real_preprocessed_valid_data'
preprocessed_test_data_path = 'data/real_preprocessed_test_data'

data_loader = DataLoader()
train_dataset = data_loader.load(train_dataset_path)
valid_dataset = data_loader.load(valid_dataset_path)
test_dataset = data_loader.load(test_dataset_path)

print("Done loading data.")

max_num_peeks = 500
def preprocess_and_save(dataset, name):
    global max_num_peeks

    def encode_to_int_mz(mz,vocab):
        mz = [vocab['SOS']] + mz + [vocab['EOS']]
        return mz

    def encode_to_normalized_intensity(data):
        record = data.reshape(-1, 1)
        scaled = (record - record.min(axis=0)) / (record.max(axis=0) - record.min(axis=0))
        data = (10000 * scaled).reshape(-1).astype(np.int64).tolist()
        data = [0] + data + [0]
        return data

    def encode_to_int_AA(sequence, vocab):
        encoded_sequence = list()
        for i in range(len(sequence)):
            if sequence[i] in vocab:
                encoded_sequence.append(vocab[sequence[i]])
        sequence = [vocab['SOS']] + encoded_sequence + [vocab['EOS']]
        return sequence

    vocab_path = './data/AA_vocab.pickle'
    save_path = '{}.tfrecords'.format(name)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(vocab)
    cnt = 0

    def _int64_feature_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    print(name)
    start = time.time()
    with tf.io.TFRecordWriter(save_path) as writer:
        for record in dataset:
            cnt+=1
            if cnt%100000 == 0 :
                print(cnt, time.time() - start)
            #score = record['score'].numpy()
            mz = np.array(record['mz'].values)
            intensity = np.array(record['intensity'].values)
            sequence = record['sequence'].numpy().decode('utf-8')

            # Pick the index list of 500 largest intensity peeks
            index_max_intensity = np.sort((-intensity).argsort()[:max_num_peeks])

            mz = mz[index_max_intensity]
            mz = mz.tolist()
            mz = encode_to_int_mz(mz,vocab)

            intensity = intensity[index_max_intensity]
            intensity = encode_to_normalized_intensity(intensity)

            sequence = list(sequence)
            sequence = encode_to_int_AA(sequence, vocab)

            feature = {
                'mz': _int64_feature_list(mz),
                'intensity': _int64_feature_list(intensity),
                'sequence': _int64_feature_list(sequence),
            }
            serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(serialized_example.SerializeToString())
    print("number of ",name, " : ", cnt)

preprocess_and_save(train_dataset, preprocessed_train_data_path)
preprocess_and_save(valid_dataset, preprocessed_valid_data_path)
preprocess_and_save(test_dataset, preprocessed_test_data_path)

'''
number of  data/real_preprocessed_train_data  :  3041294
number of  data/real_preprocessed_valid_data  :  380190
number of  data/real_preprocessed_test_data  :  390013
'''

'''
train_mz_list, train_intensity_list, train_sequence_list = prepare_data(train_dataset)
valid_mz_list, valid_intensity_list, valid_sequence_list = prepare_data(valid_dataset)
test_mz_list, test_intensity_list, test_sequence_list = prepare_data(test_dataset)

print(valid_intensity_list[:3])
#make vocabulary about train dataset
#AA_vocab = make_vocab(train_sequence_list,".AA_vocab")


vocab_path = './data/AA_vocab.pickle'

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

#encode and save train data
train_encoded_mz = encode_to_int_mz(train_mz_list, vocab)
train_encoded_intensity = encode_to_normalized_intensity(train_intensity_list)
train_encoded_sequence = encode_to_int_AA(train_sequence_list, vocab)
save_preprocessed_data(train_encoded_mz, train_encoded_intensity,
                       train_encoded_sequence, train_data_path)

#encode and save validation data
valid_encoded_mz = encode_to_int_mz(valid_mz_list, vocab)
valid_encoded_intensity = encode_to_normalized_intensity(valid_intensity_list)
valid_encoded_sequence = encode_to_int_AA(valid_sequence_list, vocab)
save_preprocessed_data(valid_encoded_mz, valid_encoded_intensity,
                       valid_encoded_sequence, validation_data_path)

print(valid_encoded_sequence[:1])
print(valid_encoded_intensity[:1])
print(valid_encoded_mz[:1])

#encode and save test data
test_encoded_mz = encode_to_int_mz(test_mz_list, vocab)
test_encoded_intensity = encode_to_normalized_intensity(test_intensity_list)
test_encoded_sequence = encode_to_int_AA(test_sequence_list, vocab)
save_preprocessed_data(test_encoded_mz, test_encoded_intensity,
                       test_encoded_sequence, test_data_path)
'''