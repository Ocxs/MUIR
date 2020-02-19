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

    dataset = 'Movie'

    self.user_id_map, self.item_id_map, \
    self.item_cate_map = self.load_map(os.path.join(args.data_dir, dataset))
    self.new_item_cate_map = {self.item_id_map[key]: self.item_cate_map[key] for key in self.item_cate_map}
    self.load_visual_feature(os.path.join(args.data_dir, dataset))


    train_data_path = os.path.join(args.data_dir, '{}/train_data.csv'.format(dataset))
    test_data_path = os.path.join(args.data_dir, '{}/test_data.csv'.format(dataset))

    if args.phase == 'train':
      self.train_pos_data, self.train_item_ids = self.load_train_data(train_data_path)
    self.test_data  = self.load_test_data(test_data_path)
    logging.info('{} test samples'.format(len(self.test_data)))


  def get_batch_data(self, idx, mode='train'):
    if mode=='train':
      batch_data = self.epoch_train_data[idx*self.batch_size: (idx+1)*self.batch_size]
    elif mode == 'test':
      if (idx+1) * self.batch_size > len(self.test_data):
        batch_data = self.test_data[-self.batch_size:]
      else:
        batch_data = self.test_data[idx * self.batch_size: (idx + 1) * self.batch_size]
    else:
      raise ValueError('invalid mode')
    user_ids, item_ids, cate_ids, labels = zip(*batch_data)
    return {
      'uid': user_ids,
      'iid': item_ids,
      'cid': cate_ids,
      'label': labels
    }

  def generate_train_data(self, neg_ratio=1.0):
    epoch_train_data = []
    for uid in range(self.n_users):
      pos_items = self.train_pos_data[uid]
      _, pos_iids, _, _ = zip(*pos_items)
      neg_iids = self.train_item_ids - set(pos_iids)

      neg_num = len(neg_iids)
      pos_num = len(pos_items)
      if neg_num < pos_num * neg_ratio:
        pass
      else:
        neg_iids = np.random.choice(list(neg_iids), size=pos_num, replace=False)
      neg_items = [(uid, iid, self.new_item_cate_map[iid], 0) for iid in neg_iids]

      epoch_train_data.extend(pos_items)
      epoch_train_data.extend(neg_items)
    random.shuffle(epoch_train_data)
    self.epoch_train_data = epoch_train_data


  def load_train_data(self, data_list_path):
    logging.info('start read data list from disk')
    with open(data_list_path, 'r') as reader:
      reader.readline()
      raw_data = map(lambda x: x.strip('\n').split(','), reader.readlines())

    data = [[] for _ in range(self.n_users)]
    train_item_ids = set()
    counter = 0
    for item in raw_data:
      train_item_ids.add(self.item_id_map[item[1]])
      if float(item[2]) > 4.5:
        counter += 1
        data[self.user_id_map[item[0]]].append((self.user_id_map[item[0]], self.item_id_map[item[1]],
                                                self.item_cate_map[item[1]], 1))

    logging.info('{} positive samples in training data'.format(counter))
    return data, train_item_ids


  def load_test_data(self, test_data_path, sep=','):
    with open(test_data_path, 'r') as reader:
      lines = map(lambda x: x.strip('\n').split(sep), reader.readlines())
      data = map(lambda x: (self.user_id_map[x[0]],
                            self.item_id_map[x[1]],
                            self.item_cate_map[x[1]],
                            int(float(x[2])> 4.5)), lines)
      data = list(data)

    return data

  def load_visual_feature(self, data_dir):
    visual_path = os.path.join(data_dir, 'cnnfv.pkl')
    visual_feature = pickle.load(open(visual_path, 'rb'), encoding='latin1')
    self.visual_feature = np.concatenate([visual_feature, [[0.0] * 4000]], axis=0)


  def load_map(self, data_dir):
    user_id_map = {}
    with open(os.path.join(data_dir, 'user_id_map.txt'), 'r') as reader:
      for line in reader:
        arr = line.strip('\n').split(':')
        user_id_map[arr[0]] = int(arr[1])

    item_id_map = {}
    with open(os.path.join(data_dir, 'item_id_map.txt'), 'r') as reader:
      for line in reader:
        arr = line.strip('\n').split(':')
        item_id_map[arr[0]] = int(arr[1])

    item_cate_map = {}
    with open(os.path.join(data_dir, 'item_cate_map.txt'), 'r') as reader:
      for line in reader:
        arr = line.strip('\n').split(':')
        item_cate_map[arr[0]] = [int(item) for item in arr[1].split(',')]


    return user_id_map, item_id_map, item_cate_map

  def del_temp(self):
    del self.visual_feature