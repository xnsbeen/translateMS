import tensorflow as tf
import time
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

size_train_dataset = 100000
train_dataset = tf.data.TFRecordDataset(path_train_data).take(size_train_dataset)
valid_dataset = tf.data.TFRecordDataset(path_valid_data).map(parse_function)

#Load vocabulary
with open('./data/mz_vocab.pickle','rb') as f:
    mz_vocab = pickle.load(f)
with open('./data/AA_vocab.pickle','rb') as f:
    AA_vocab = pickle.load(f)

#Get vocabulary size
mz_vocab_size = len(mz_vocab)+1
AA_vocab_size = len(AA_vocab)+1

print(f'mz vocab size : {mz_vocab_size}, AA vocab size : {AA_vocab_size}')

#Set batchs
BATCH_SIZE = 1024
NUM_BATCHS = int(size_train_dataset/BATCH_SIZE)
train_batches = (train_dataset
                 .map(parse_function)
                 .padded_batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))


def create_masks(input, target):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(input)
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


'''
d_model : input(embedding), ouput 차원
num_layers : 인코더, 디코더 층
num_heads : 멀티헤드 수
d_ff : feedforward 차원 
'''
D_MODEL = 64
NUM_LAYERS = 2
NUM_HEADS = 2
DFF = 128
DROPOUT_RATE = 0.2

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=mz_vocab_size,
    target_vocab_size=AA_vocab_size,
    positional_encoding_input = 500000,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

#save checkpoint
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(input, target):

    target_input = target[:, :-1]
    target_real = target[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(input, target_input,
                                   True,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)

        loss = loss_function(target_real, predictions)


    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, predictions))


def evaluate_aminoacid_level(dataset):
    batch_size = 500
    num_batchs = 0
    accuracy = 0
    dataset_batchs = dataset.padded_batch(batch_size = batch_size, drop_remainder=True)

    for batch, (input, target) in enumerate(dataset_batchs):
        num_batchs = batch+1

        target_input = target[:, :-1]
        target_real = target[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

        predictions, _ = transformer(input, target_input,
                                   False,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)

        accuracy += accuracy_function(target_real, predictions)

    return accuracy/num_batchs


EPOCHS = 30
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (input, target) in enumerate(train_batches):
        train_step(input, target)

        if (batch+1)%10 == 0 :
            print(f'Epoch {epoch + 1} Batch {batch+1}/{NUM_BATCHS} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print(f'Accuracy of validation data for amino acid level : {evaluate_aminoacid_level(valid_dataset):.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

