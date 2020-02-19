#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# __date__ = 17-10-24:20-08
# __author__ = Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import random, logging, os
import pickle



class DataLoader(object):
  def __init__(self, args):
    self.data_dir = args.data_dir
    self.batch_size = args.batch_size
    random.seed(args.seed)
    self.n_users = args.n_users
    self.n_items = args.n_items
    self.n_cates = args.n_cates

    train_data_path = os.path.join(args.data_dir, 'Video/train_data.csv')
    test_data_path = os.path.join(args.data_dir, 'Video/test_data.csv')

    self.load_data_from_disk()
    if args.phase == 'train':
      self.train_data = self.load_train_data(train_data_path)
    self.test_data = self.load_test_data(test_data_path)
    logging.info('{} test samples'.format(len(self.test_data)))


  def get_train_batch(self, idx):
    batch_data = self.epoch_train_data[idx*self.batch_size: (idx+1)*self.batch_size]
    user_ids, item_ids, cate_ids, labels = zip(*batch_data)
    return {
      'uid': user_ids,
      'iid': item_ids,
      'label': labels,
      'cid': cate_ids
    }


  def get_test_batch(self, idx):
    if (idx + 1) * self.batch_size > len(self.test_data):
      batch_data = self.test_data[-self.batch_size:]
    else:
      batch_data = self.test_data[idx * self.batch_size: (idx + 1) * self.batch_size]

    user_ids, item_ids, cate_ids, labels = zip(*batch_data)
    item_vecs = self.get_test_visual_feature(item_ids)
    return {
      'uid': user_ids,
      'iid': item_ids,
      'label': labels,
      'cid': cate_ids,
      'item_vec': item_vecs
    }


  def generate_train_data(self, neg_ratio=1.0):
    logging.info('generate samples for training')
    epoch_train_data = []
    for item in self.train_data:
      pos_num, neg_num = len(item[0]), len(item[1])
      epoch_train_data.extend(item[0])
      if neg_num < pos_num * neg_ratio:
        epoch_train_data.extend(item[1])
      else:
        epoch_train_data.extend(random.sample(item[1], int(pos_num*neg_ratio)))

    random.shuffle(epoch_train_data)
    self.epoch_train_data = epoch_train_data
    logging.info('sampling {} train samples in this epoch'.format(len(self.epoch_train_data)))


  def load_train_data(self, data_list_path):
    logging.info('start read data list from disk')
    with open(data_list_path, 'r') as reader:
      reader.readline()
      raw_data = map(lambda x: x.strip('\n').split(','), reader.readlines())

    data = [[[], []] for _ in range(self.n_users)]
    for item in raw_data:
      if int(item[3]) == 1:
        data[int(item[0])][0].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
      else:
        data[int(item[0])][1].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    return data

  def load_test_data(self, test_data_path, sep=','):
    with open(test_data_path, 'r') as reader:
      reader.readline()
      lines = map(lambda x: x.strip('\n').split(sep), reader.readlines())
      data = map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3])), lines)

    return list(data)


  def get_test_visual_feature(self, vids):
    return [self.test_visual_feature[i] for i in vids]


  def load_data_from_disk(self):
    train_visual_path = os.path.join(self.data_dir, 'Video/train_cover_image_feature.npy')
    test_visual_path = os.path.join(self.data_dir, 'Video/test_cover_image_feature.npy')

    logging.info('load train visual feature')
    train_visual_feature = np.load(train_visual_path)
    self.train_visual_feature = np.concatenate([train_visual_feature, [[0.0]*512]], axis=0)

    logging.info('load test visual feature')
    self.test_visual_feature = np.load(test_visual_path)




  def del_temp(self):
    del self.train_visual_feature