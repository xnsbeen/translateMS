import time
import pickle
import numpy as np
import tensorflow as tf

from data.msms import DataLoader
from data.msms import make_vocab
dataset_path = 'data/theoretical_raw_msms_data.tfrecords'
train_data_path = 'data/theoretical_preprocessed_train_data'
valid_data_path = 'data/theoretical_preprocessed_valid_data'
test_data_path = 'data/theoretical_preprocessed_test_data'

data_loader = DataLoader()
dataset = data_loader.load(dataset_path)

#real_data_size = 256105
#theoretical_data_size = 7448762
record_size = 7448762
train_dataset, valid_dataset, test_dataset \
    = data_loader.split(dataset, record_size=record_size, train_prop=0.99, valid_prop=0.005)
print("Done loading data.")

max_num_peeks = 0

def preprocess_and_save(dataset, name):
    global max_num_peeks

    def encode_to_int_mz(mz,vocab):
        mz = [vocab['SOS']] + mz + [vocab['EOS']]
        return mz

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

    start = time.time()
    with tf.io.TFRecordWriter(save_path) as writer:
        for record in dataset:
            cnt+=1
            if cnt%10000 == 0 :
                print(cnt, time.time() - start)
            #score = record['score'].numpy()
            mz = np.array(record['mz'].values)
            intensity = np.array(record['intensity'].values)
            sequence = record['sequence'].numpy().decode('utf-8')

            # Pick the index list of 500 largest intensity peeks
            #index_max_intensity = np.sort((-intensity).argsort()[:max_num_peeks])

            mz = mz.tolist()
            mz = encode_to_int_mz(mz,vocab)
            max_num_peeks = max(max_num_peeks,len(mz))
            #intensity = intensity[index_max_intensity]
            sequence = list(sequence)
            sequence = encode_to_int_AA(sequence, vocab)
            #intensity_list.append(intensity.tolist())

            feature = {
                'mz': _int64_feature_list(mz),
                # 'intensity': _int64_feature_list(intensity),
                'sequence': _int64_feature_list(sequence),
            }
            serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(serialized_example.SerializeToString())

#make_vocab(train_dataset, './data/AA_vocab')
preprocess_and_save(train_dataset, train_data_path)
print('Complete train data')
preprocess_and_save(valid_dataset, valid_data_path)
print('Complete valid data')
preprocess_and_save(test_dataset, test_data_path)
print('Complete test data')


print("max number of mz peeks", max_num_peeks)




