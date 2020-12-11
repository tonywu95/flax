import os.path as osp
import tensorflow as tf
from tensorflow.keras import preprocessing

batch_size = 32
seed = 42

parent_dir = "copy_data"
raw_train_src = tf.data.TextLineDataset(osp.join(parent_dir, "train.src"))
raw_train_tgt = tf.data.TextLineDataset(osp.join(parent_dir, "train.tgt"))
raw_train = tf.data.Dataset.zip((raw_train_src, raw_train_tgt))

#raw_valid = tf.data.TextLineDataset(osp.join(parent_dir, "valid"))
#raw_test = tf.data.TextLineDataset(osp.join(parent_dir, "test"))

for text_batch, label_batch in raw_train.take(100):
    print("Question: ", text_batch.numpy())
    print("Label:", label_batch.numpy())
import pdb;pdb.set_trace()
