import tensorflow as tf
import time

from transformer.model import Transformer
from transformer.model import ModifiedTransformer
from transformer.model import create_padding_mask
from transformer.model import create_look_ahead_mask
from transformer.optimizer import CustomSchedule

feature_description = {
            'sequence': tf.io.VarLenFeature(tf.int64),
            'intensity': tf.io.VarLenFeature(tf.int64),
            'mz': tf.io.VarLenFeature(tf.int64),
            }

def parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto,feature_description)
    mz = parsed_example['mz'].values
    intensity = parsed_example['intensity'].values
    sequence = parsed_example['sequence'].values
    return mz, intensity, sequence

path_train_data='./data/real_preprocessed_train_data.tfrecords'
path_valid_data='./data/real_preprocessed_valid_data.tfrecords'

#Number of full data : 256105
size_train_dataset = 300
size_valid_dataset = 200
train_dataset = tf.data.TFRecordDataset(path_train_data).take(size_train_dataset)
valid_dataset = tf.data.TFRecordDataset(path_valid_data).take(size_valid_dataset).map(parse_function)

#Set batchs
BATCH_SIZE = 64
NUM_BATCHS = int(size_train_dataset/BATCH_SIZE)
train_batches = (train_dataset
                 .map(parse_function)
                 .padded_batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

print(train_batches)

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

pretrained_transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=600000,
    target_vocab_size=30,
    positional_encoding_input = 1000,
    positional_encoding_target = 50,
    dropout_rate=DROPOUT_RATE)

modified_transformer = ModifiedTransformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    intensity_vocab_size=12000,
    target_vocab_size=30,
    dropout_rate=DROPOUT_RATE)

#save checkpoint
checkpoint_path_pretraining = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=pretrained_transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path_pretraining, max_to_keep=10)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(input, intensity, target):

    target_input = target[:, :-1]
    target_real = target[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

    with tf.GradientTape() as tape:
        #return : enc_output, dec_output, final_output, attention_weights
        encoder1_output, decoder1_output, _, _ = \
            pretrained_transformer(input,
                                   target_input,
                                   False,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)

        predictions, _ = modified_transformer(encoder1_output,
                                             intensity,
                                             decoder1_output,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

        loss = loss_function(target_real, predictions)

    gradients = tape.gradient(loss, modified_transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, modified_transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(target_real, predictions))


def evaluate_aminoacid_level(dataset):
    batch_size = 200
    num_batchs = 0
    accuracy = 0
    loss = 0
    dataset_batchs = dataset.padded_batch(batch_size = batch_size, drop_remainder=True)

    for batch, (input, intensity, target) in enumerate(dataset_batchs):
        num_batchs = batch+1
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

        encoder1_output, decoder1_output, _, _ = \
            pretrained_transformer(input,
                                   target_input,
                                   False,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)

        predictions, _ = modified_transformer(encoder1_output,
                                              intensity,
                                              decoder1_output,
                                              False,
                                              enc_padding_mask,
                                              combined_mask,
                                              dec_padding_mask)

        loss += loss_function(target_real, predictions)
        accuracy += accuracy_function(target_real, predictions)

    return loss/num_batchs, accuracy/num_batchs


EPOCHS = 30
epoch = 0
save_checkpoint_path = './checkpoints/transfer_learning'
save_ckpt = tf.train.Checkpoint(transformer=modified_transformer,
                           optimizer=optimizer)

save_ckpt_manager = tf.train.CheckpointManager(save_ckpt, save_checkpoint_path, max_to_keep=10)
if save_ckpt_manager.latest_checkpoint:
    save_ckpt.restore(save_ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

#for epoch in range(EPOCHS):
while True:
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (input, intensity, target) in enumerate(train_batches):
        train_step(input, intensity, target)

        print('\r',f'Epoch {epoch + 1} | batch {batch+1}/{NUM_BATCHS} Loss {train_loss.result():.3f} Accuracy {train_accuracy.result():.4f}',end='')

    ckpt_save_path = save_ckpt_manager.save()
    print('\r', f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print('\r',f'Epoch {epoch + 1} : Time {time.time() - start:.2f}s')
    print(f'\tTrain | Loss {train_loss.result():.3f}, Accuracy {train_accuracy.result():.3f}')
    valid_loss, valid_accuracy = evaluate_aminoacid_level(valid_dataset)
    print(f'\tValid | Loss {valid_loss:.3f}, Accuracy {valid_accuracy:.3f}')

    epoch+=1

