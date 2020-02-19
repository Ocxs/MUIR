#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import time

import tensorflow as tf
import numpy as np

from data_loader import DataLoader


class Solver(object):
  def __init__(self, model, args):
    self.model = model
    self.phase = args.phase
    self.max_epoch = args.max_epoch
    self.batch_size = args.batch_size
    self.display = args.display
    self.lr = args.lr
    self.out_dir = args.out_dir
    self.model_dir = os.path.join(self.out_dir, 'ckpt')
    self.neg_ratio = args.neg_ratio
    self.emb_dim = args.emb_dim

    sub_dir = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
      args.batch_size, args.lr, args.optimizer, args.w_reg, args.emb_reg, args.dropout_keep,
      args.neg_ratio, args.emb_dim, args.interaction, '-'.join([str(i) for i in args.agg_layers])
    )
    self.model_dir = os.path.join(self.model_dir, sub_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    self.data= DataLoader(args)


  def create_model(self, sess):
    sess.run(tf.global_variables_initializer())
    params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
    saver = tf.train.Saver(params, max_to_keep=1)
    self.model.restore_visual_feature(self.data.visual_feature, sess)
    self.data.del_temp()


    if self.phase == 'test':
      has_ckpt = tf.train.get_checkpoint_state(self.model_dir)
      if has_ckpt:
        model_path = has_ckpt.model_checkpoint_path
        saver.restore(sess, model_path)
        logging.info("Load model from {}".format(model_path))
      else:
        raise ValueError("No checkpoint file found in {}".format(self.model_dir))
    else:
      logging.info('Create model with fresh parameters')


  def train(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      self.create_model(sess)

      min_loss = 1e5
      early_stop = 0
      max_test_ndcg, max_test_auc = 0, 0
      best_epoch = 0
      lr = self.lr
      test_auc_lst, lr_lst, loss_lst = [], [], []
      top_k_p, top_k_r, top_k_ndcg = [], [], []

      for epoch in range(self.max_epoch):
        self.data.generate_train_data(self.neg_ratio)
        length = len(self.data.epoch_train_data)
        loop_num = int(length // self.batch_size)


        epoch_avg_loss = 0.0

        start = time.time()
        for i in range(loop_num):
          batch_data = self.data.get_batch_data(i, 'train')
          inference_loss, reg_loss, acc = self.model.train(sess, batch_data, lr)
          epoch_avg_loss += inference_loss

        run_time = time.time() - start
        epoch_avg_loss /= loop_num
        logging.info('epoch: {}- train loss: {:.4f} in {:.2f}s, lr: {:.8f}, {} iters, {} train samples'
                     .format(epoch+1, epoch_avg_loss, run_time, lr, loop_num, length))

        if epoch_avg_loss < min_loss:
          if min_loss - epoch_avg_loss < 0.005:
            early_stop += 1
            lr *= 0.5
          else:
            early_stop = max(0, early_stop - 1)
          min_loss = epoch_avg_loss
        else:
          early_stop += 1
          lr *= 0.5

        test_p_k, test_r_k, test_ndcg_k, test_auc = self.eval(sess, epoch)

        if test_auc > max_test_auc:
          max_test_auc = test_auc
          save_path = os.path.join(self.model_dir, 'model-{:.4f}-{:.4f}.ckpt'.format(max_test_auc, test_r_k[0]))
          self.model.save(sess, save_path, epoch + 1)
          best_epoch = epoch + 1


        test_auc_lst.append(test_auc)
        lr_lst.append(lr)
        loss_lst.append(epoch_avg_loss)
        top_k_p.append(test_p_k)
        top_k_r.append(test_r_k)
        top_k_ndcg.append(test_ndcg_k)

        if early_stop > 5:
          break

      logging.info('loss     list: [{}], min: {:.4f}, {}'.format(','.join(['{:.4f}'.format(i) for i in loss_lst]),
                                                             min(loss_lst), np.argmin(loss_lst)+1))
      logging.info('test auc list: [{}], max: {:.4f}, {}'.format(','.join(['{:.4f}'.format(i) for i in test_auc_lst]),
                                                             max(test_auc_lst), np.argmax(test_auc_lst)+1))
      logging.info('lr       list: [{}], min: {:.8f}, {}'.format(','.join(['{:.8f}'.format(i) for i in lr_lst]),
                                                             min(lr_lst), np.argmin(lr_lst)+1))

      logging.info('top                  10      20     50     100 ')
      for idx, item in enumerate(top_k_p):
        logging.info('epoch {}: precision: [{}]'.format(idx+1, ','.join(['{:.4f}'.format(i) for i in item])))
      logging.info('----')
      for idx, item in enumerate(top_k_r):
        logging.info('epoch {}: recall: [{}]'.format(idx + 1, ','.join(['{:.4f}'.format(i) for i in item])))
      logging.info('----')
      for idx, item in enumerate(top_k_ndcg):
        logging.info('epoch {}: ndcg: [{}]'.format(idx + 1, ','.join(['{:.4f}'.format(i) for i in item])))
      logging.info('best epoch: {}'.format(best_epoch))


  def eval(self, sess, epoch, save_pred=False):
    length = len(self.data.test_data)
    loop_num = int(length // self.batch_size) + 1
    pred_dict = {}
    test_loss = 0.0

    start = time.time()
    for step in range(loop_num):
      batch_data = self.data.get_batch_data(step, 'test')
      loss, logits, _ = self.model.test(sess, batch_data)
      test_loss += loss
      for i in range(self.batch_size):
        if pred_dict.get(batch_data['uid'][i]) is None:
          pred_dict[batch_data['uid'][i]] = []
        pred_dict[batch_data['uid'][i]].append([logits[i], int(batch_data['label'][i]), int(batch_data['iid'][i])])
    end_time = time.time() - start

    if save_pred:
      np.save(os.path.join(self.out_dir, 'LIN_Movie_pred{}.npy'.format(self.emb_dim)), pred_dict)

    test_loss /= loop_num
    p_k, r_k, ndcg_k = [], [], []
    for top_k in [5, 10, 20, 50]:
      precision, recall, ndcg  = compute_p_and_r_and_ndcg(pred_dict, top_k)
      p_k.append(precision)
      r_k.append(recall)
      ndcg_k.append(ndcg)

    auc = compute_auc(pred_dict)
    logging.info('epoch: {}- test  loss: {:.4f} in {:.2f}s, auc:{:.4f}, precision:{:.4f}, '
                 'recall:{:.4f}, ndcg:{:.4f} in top10 , {} samples'.format(epoch+1, test_loss, end_time, auc, p_k[0],
                                                                          r_k[0], ndcg_k[0], length))
    return p_k, r_k, ndcg_k, auc




  def test(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=8,
                            intra_op_parallelism_threads=8,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      self.create_model(sess)
      logging.info('start test phase')
      test_p_k, test_r_k, test_ndcg_k, test_auc = self.eval(sess, 1, True)
      logging.info('precision: [{}]'.format(','.join(['{:.4f}'.format(i) for i in test_p_k])))
      logging.info('recall: [{}]'.format(','.join(['{:.4f}'.format(i) for i in test_r_k])))
      logging.info('ndcg: [{}]'.format(','.join(['{:.4f}'.format(i) for i in test_ndcg_k])))
      logging.info('test auc: {:.4f}'.format(test_auc))


def compute_p_and_r_and_ndcg(pred_dict, top_k=10):
  """
  compute precision and recall
  """
  precisions, recalls, ndcgs = [], [], []
  for key in pred_dict:
    preds = pred_dict[key]
    preds.sort(key=lambda x: x[0], reverse=True)
    preds = np.array(preds)
    # precision and recall
    precisions.append([sum(preds[:top_k, 1]) / top_k, len(preds)])
    recalls.append([sum(preds[:top_k, 1]) / sum(preds[:, 1]), len(preds)])

    pos_idx = np.where(preds[:top_k, 1] == 1)[0]
    dcg = np.sum(np.log(2) / np.log(2 + pos_idx))
    idcg = np.sum(np.log(2) / np.log(2 + np.arange(len(pos_idx))))
    ndcg = dcg / (idcg + 1e-8)
    ndcgs.append([ndcg, len(preds[:top_k])])

  precisions = np.array(precisions)
  p = sum(precisions[:, 0] * precisions[:, 1]) / sum(precisions[:, 1])
  recalls = np.array(recalls)
  r = sum(recalls[:, 0] * recalls[:, 1]) / sum(recalls[:, 1])
  ndcgs = np.array(ndcgs)
  ndcg = sum(ndcgs[:, 0] * ndcgs[:, 1]) / sum(ndcgs[:, 1])
  return p, r, ndcg

def compute_auc(pred_dict):
  auc_lst = []
  for key in pred_dict:
    preds = pred_dict[key]
    preds.sort(key=lambda x: x[0], reverse=True)
    preds = np.array(preds)
    pos_num = sum(preds[:, 1])
    neg_num = len(preds) - pos_num
    if pos_num == 0 or neg_num == 0:
      continue
    # auc
    pos_count, neg_count = 0, 0
    for i in range(len(preds)):
      if preds[i, 1] == 0:
        neg_count += (pos_num - pos_count)
      else:
        pos_count += 1
      if pos_count == pos_num:
        auc = 1 - (neg_count / (pos_num * neg_num))
        auc_lst.append([auc, len(preds), key])
        break
  auc_lst = np.array(auc_lst)
  auc = sum(auc_lst[:, 0] * auc_lst[:, 1]) / sum(auc_lst[:, 1])
  return auc