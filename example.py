import tensorflow as tf
import time
import numpy as np
import pickle

from transformer.model import Transformer
from transformer.model import create_padding_mask
from transformer.model import create_look_ahead_mask
from transformer.optimizer import CustomSchedule

a = tf.convert_to_tensor([1,2,3], dtype=tf.int64)
b = tf.convert_to_tensor([1,1,3], dtype=tf.int64)

print(a==b)

if tf.reduce_all(a==b):
    print("equal!")