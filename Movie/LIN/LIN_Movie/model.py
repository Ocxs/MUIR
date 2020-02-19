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
from nets import dot_attention



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
    self.n_users = args.n_users
    self.n_items = args.n_items
    self.n_cates = args.n_cates

    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    with tf.device('/gpu:0'):
      self.init_embedding()
      self.set_placeholder()
      self.set_optimizer()
      with tf.variable_scope('LIN'):
        self.train_inference()
        tf.get_variable_scope().reuse_variables()
        self.test_inference()



  def train_inference(self):
    inference_loss, reg_loss, acc, _ = self.build_model(
      self.user_ids_ph,
      self.item_ids_ph,
      self.cate_ids_ph,
      self.labels_ph,
      self.dropout_keep,
      'train'
    )

    train_params = tf.trainable_variables()
    # train op
    gradients = tf.gradients(inference_loss + reg_loss, train_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, train_params), self.global_step)
    self.inference_loss = inference_loss
    self.reg_loss = reg_loss
    self.train_acc = acc

    # saver
    params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
    self.saver = tf.train.Saver(params, max_to_keep=1)


  def test_inference(self):
    loss, acc, logits = self.build_model(
      self.user_ids_ph,
      self.item_ids_ph,
      self.cate_ids_ph,
      self.labels_ph,
      1.0,
      'test'
    )

    self.test_loss = loss
    self.test_acc = acc
    self.test_logits = tf.nn.sigmoid(logits)


  def build_model(self,
                  user_ids,
                  item_ids,
                  cate_ids,
                  labels,
                  keep_prob,
                  phase):

    with tf.variable_scope('feature_representation'):
      # user
      user_emb = tf.nn.embedding_lookup(self.user_embedding, user_ids)
      latent_interest = user_emb
      # item
      cate_emb = tf.reduce_mean(tf.nn.embedding_lookup(self.item_category_embedding, cate_ids), axis=1)

      visual_feat = tf.nn.embedding_lookup(self.visual_feature, item_ids)
      visual_emb = dense(visual_feat, self.emb_dim, ['w1'], 1.0)

      item_emb = visual_emb + cate_emb


    with tf.variable_scope('aggregation'):
      if self.interaction == 'inner':
        fusion_input = latent_interest * item_emb
        logits = tf.reduce_sum(fusion_input, axis=1)
      else:
        if self.interaction == 'product':
          fusion_input = latent_interest * item_emb
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
      emb_l2_loss = tf.nn.l2_loss(cate_emb) + tf.nn.l2_loss(user_emb)
      w_l2_loss_lst = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name]
      if len(w_l2_loss_lst) != 0:
        w_l2_loss = tf.add_n(w_l2_loss_lst)
        reg_loss = w_l2_loss * self.w_reg + emb_l2_loss * self.emb_reg
      else:
        reg_loss = emb_l2_loss * self.emb_reg
      return inference_loss, reg_loss, acc, logits
    else:
      return inference_loss, acc, logits


  def train(self, sess, data, lr):
    feed_dicts = {
      self.user_ids_ph: data['uid'],
      self.item_ids_ph: data['iid'],
      self.cate_ids_ph: data['cid'],
      self.labels_ph: data['label'],
      self.lr_ph: lr
    }
    train_run_op = [self.inference_loss, self.reg_loss, self.train_acc, self.train_op]
    inference_loss, reg_loss, acc, _ = sess.run(train_run_op, feed_dicts)
    return inference_loss, reg_loss, acc


  def test(self, sess, data):
    feed_dicts = {
      self.user_ids_ph: data['uid'],
      self.item_ids_ph: data['iid'],
      self.cate_ids_ph: data['cid'],
      self.labels_ph: data['label']

    }
    test_run_op = [self.test_loss, self.test_logits, self.test_acc]
    loss, logits, acc = sess.run(test_run_op, feed_dicts)
    return loss, logits, acc


  def save(self, sess, model_path, epoch):
    self.saver.save(sess, model_path, global_step=epoch)


  def compute_acc(self, logit, labels):
    pred = tf.cast(tf.nn.sigmoid(logit)>=0.5, tf.float32)
    correct_pred = tf.equal(pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


  def set_placeholder(self):
    self.user_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.item_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.cate_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, None))
    self.labels_ph = tf.placeholder(tf.float32, shape=(self.batch_size,))
    self.lr_ph = tf.placeholder(tf.float32, shape=())


  def restore_visual_feature(self, visual_feature, sess):
    visual_ph = tf.placeholder(tf.float32, [self.n_items+1, 4000])
    emb_init = self.visual_feature.assign(visual_ph)
    sess.run(emb_init, feed_dict={visual_ph: visual_feature})
    logging.info('load train visual feature into GPU memory successfully')


  def init_embedding(self):
    self.user_embedding = var_init('user_embedding', [self.n_users, self.emb_dim],
                                   tf.random_normal_initializer(mean=0.0, stddev=0.01))

    item_category_embedding = var_init('item_category_embedding', [self.n_cates, self.emb_dim],
                                       tf.random_normal_initializer(mean=0.0, stddev=0.01))
    self.item_category_embedding = tf.concat([item_category_embedding, tf.zeros((1, self.emb_dim))], axis=0)

    self.visual_feature = tf.Variable(tf.constant(0.0, shape=[self.n_items+1, 4000]), trainable=False,
                                      name='visual_feat')



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


















