#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from nets import var_init, dense

from nets import dnn, mlp
from nets import neighbor_attention
from nets import item_interest_attention



class Model(object):
  def __init__(self, args):
    self.batch_size = args.batch_size
    self.w_reg = args.w_reg
    self.emb_reg = args.emb_reg
    self.dropout_keep = args.dropout_keep
    self.agg_layers = args.agg_layers
    self.interaction = args.interaction
    self.optimizer = args.optimizer
    self.emb_dim = args.emb_dim
    self.max_gradient_norm = args.max_gradient_norm
    self.seed = args.seed
    self.neighbor_length = args.neighbor_length
    self.n_users = args.n_users
    self.n_items = args.n_items
    self.n_cates = args.n_cates
    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    with tf.device('/gpu:0'):
      self.init_embedding()
      self.set_placeholder()
      self.set_optimizer()
      with tf.variable_scope('NIN'):
        self.train_inference()
        tf.get_variable_scope().reuse_variables()
        self.test_inference()



  def train_inference(self):
    item_vec = tf.nn.embedding_lookup(self.train_visual_embedding, self.item_ids_ph)
    inference_loss, l2_w_loss, l2_emb_loss, acc, _ = self.build_model(self.user_ids_ph,
                                                                      item_vec,
                                                                      self.cate_ids_ph,
                                                                      self.labels_ph,
                                                                      self.neighbor_user_ids_ph,
                                                                      self.dropout_keep,
                                                                      'train')

    train_params = tf.trainable_variables()
    # train op
    gradients = tf.gradients(inference_loss + l2_w_loss + l2_emb_loss, train_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, train_params), self.global_step)
    self.inference_loss = inference_loss
    self.l2_w_loss = l2_w_loss
    self.l2_emb_loss = l2_emb_loss
    self.train_acc = acc

    # saver
    params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
    self.saver = tf.train.Saver(params, max_to_keep=1)


  def test_inference(self):
    loss, acc, logits = self.build_model(self.user_ids_ph,
                                         self.item_vec_ph,
                                         self.cate_ids_ph,
                                         self.labels_ph,
                                         self.neighbor_user_ids_ph,
                                         1.0,
                                         'test')

    self.test_loss = loss
    self.test_acc = acc
    self.test_logits = tf.nn.sigmoid(logits)


  def build_model(self,
                  user_ids,
                  item_vec,
                  cate_ids,
                  labels,
                  neighbor_user_ids,
                  keep_prob,
                  phase):

    with tf.variable_scope('feature_representation'):
      # user
      neighbor_user_emb = tf.nn.embedding_lookup(self.user_embedding, neighbor_user_ids)

      # item
      item_emb = dense(item_vec, self.emb_dim, ['w1'], keep_prob)
      cate_emb = tf.nn.embedding_lookup(self.item_category_embedding, cate_ids)

      item_emb = item_emb + cate_emb

    with tf.variable_scope('neighborhood_level_interest_network'):
      avg_neighbor_emb = tf.reduce_mean(neighbor_user_emb, axis=1, keep_dims=True)
      neighbor_emb = neighbor_attention(neighbor_user_emb, item_emb, avg_neighbor_emb, 1.0, 'relu', False)

    # ef: early fusion
    # if: intermediate fusion
    # lf: late fusion
    with tf.variable_scope('interaction'):
      if self.interaction == 'inner':
        logits = tf.reduce_sum(neighbor_emb*item_emb, axis=1)
      else:
        if self.interaction == 'product':
          fusion_input = neighbor_emb * item_emb
        else:
          raise ValueError('invalid fusion way: {}'.format(self.interaction))

        with tf.variable_scope('aggregation_layer'):
          logits = dnn(fusion_input, self.agg_layers, keep_prob)

    inference_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=labels)
    )

    acc = self.compute_acc(logits, self.labels_ph)

    if phase == 'train':
      emb_l2_loss = tf.nn.l2_loss(neighbor_user_emb) + tf.nn.l2_loss(cate_emb)

      w_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name])
      l2_w_loss = w_l2_loss * self.w_reg
      l2_emb_loss = emb_l2_loss * self.emb_reg
      return inference_loss, l2_w_loss, l2_emb_loss, acc, logits
    else:
      return inference_loss, acc, logits


  def train(self, sess, data, lr):
    feed_dicts = {
      self.user_ids_ph: data['uid'],
      self.item_ids_ph: data['iid'],
      self.labels_ph: data['label'],
      self.neighbor_user_ids_ph: data['nuid'],
      self.cate_ids_ph: data['cid'],
      self.lr_ph: lr
    }
    train_run_op = [self.inference_loss, self.l2_w_loss, self.l2_emb_loss,
                    self.train_acc, self.train_op]
    inference_loss, l2_w_loss, l2_emb_loss, acc, _ = sess.run(train_run_op, feed_dicts)
    return inference_loss, l2_w_loss, l2_emb_loss, acc


  def test(self, sess, data):
    feed_dicts = {
      self.user_ids_ph: data['uid'],
      self.item_ids_ph: data['iid'],
      self.labels_ph: data['label'],
      self.neighbor_user_ids_ph: data['nuid'],
      self.cate_ids_ph: data['cid'],
      self.item_vec_ph: data['item_vec'],

    }
    test_run_op = [self.test_loss, self.test_logits, self.test_acc]
    loss, logits, acc = sess.run(test_run_op, feed_dicts)
    return loss, logits, acc


  def save(self, sess, model_path, epoch):
    self.saver.save(sess, model_path, global_step=epoch)
    logging.info("Saved model in epoch {}".format(epoch))


  def compute_acc(self, logit, labels):
    pred = tf.cast(tf.nn.sigmoid(logit)>=0.5, tf.float32)
    correct_pred = tf.equal(pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


  def set_placeholder(self):
    self.user_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.item_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.cate_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.labels_ph = tf.placeholder(tf.float32, shape=(self.batch_size,))
    self.neighbor_user_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.neighbor_length))
    self.item_vec_ph = tf.placeholder(tf.float32, shape=(self.batch_size, 512))
    self.lr_ph = tf.placeholder(tf.float32, shape=())


  def restore_train_visual_feature(self, visual_feature, sess):
    visual_ph = tf.placeholder(tf.float32, [self.n_items+1, 512])
    emb_init = self.train_visual_embedding.assign(visual_ph)
    sess.run(emb_init, feed_dict={visual_ph: visual_feature})
    logging.info('load train visual feature into GPU memory successfully')



  def init_embedding(self):
    self.user_embedding = var_init('user_embedding', [self.n_users, self.emb_dim],
                                   tf.random_normal_initializer(mean=0.0, stddev=0.01))

    self.train_visual_embedding = tf.Variable(tf.constant(0.0, shape=[self.n_items+1, 512]), trainable=False,
                                              name='train_visual_emb')


    self.item_category_embedding = var_init('item_category_embedding', [self.n_cates, self.emb_dim],
                                       tf.random_normal_initializer(mean=0.0, stddev=0.01))


  def set_optimizer(self):
    if self.optimizer == 'sgd':
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr_ph)
    elif self.optimizer == 'adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
    elif (self.optimizer == 'adadelta'):
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr_ph)
    elif (self.optimizer == 'adagrad'):
      self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr_ph, initial_accumulator_value=0.9)
    elif (self.optimizer == 'rms'):
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr_ph, decay=0.9, epsilon=1e-6)
    elif (self.optimizer == 'moment'):
      self.opt = tf.train.MomentumOptimizer(self.lr_ph.learn_rate, 0.9)
    else:
      raise ValueError('do not support {} optimizer'.format(self.optimizer))
























