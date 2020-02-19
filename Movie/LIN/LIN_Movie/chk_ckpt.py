#_*_coding:utf-8_*_
#author: Xusong Chen
#date: 3/18/19 11:03 AM

import os
import tensorflow as tf
data_dir = '/home/chen/Downloads/v7_5/LIN_Movie/ckpt/2048_0.001_adam_1e-05_1e-07_0.85_1.0_8_product_32_epoch_decay'

with tf.device('/gpu:0'):
  config = tf.ConfigProto(allow_soft_placement=True)

  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  new_saver = tf.train.import_meta_graph(os.path.join(data_dir, 'model-1.0000.ckpt-1.meta'))
  sess.run(tf.global_variables_initializer())
  #print( tf.train.latest_checkpoint('./'))
  what=new_saver.restore(sess, os.path.join(data_dir, 'model-1.0000.ckpt-1'))
  #print(what)

  all_vars = tf.trainable_variables()#tf.get_collection('parameters')
  print(all_vars)
  for v in all_vars:
      v_ = sess.run(v)
      print(v_)