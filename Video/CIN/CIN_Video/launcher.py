#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import time
import os
import logging
import sys

from model import Model
from solver import Solver

import tensorflow as tf


def process_args(args):
  def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
  parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
  parser.add_argument('--max-epoch', type=int, default=2, help='the number of epochs')
  parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
  parser.add_argument('--w-reg', type=float, default=5e-5, help='regularization coefficient for network parameters')
  parser.add_argument('--emb-reg', type=float, default=5e-5, help='regularization coefficient for embeddings')
  parser.add_argument('--dropout-keep', type=float, default=0.7, help='The probability that each element is kept.')

  parser.add_argument('--neg-ratio', type=float, default=1.0, help='sample ratio')
  parser.add_argument('--emb-dim', type=int, default=64, help='dim of embedding')
  parser.add_argument('--interaction', type=str, default='product', help='interaction method')
  parser.add_argument('--agg-layers', type=int, nargs='+', default=[128], help='aggregation layer')

  parser.add_argument('--max-gradient-norm', type=float, default=5.0)
  parser.add_argument('--display', type=int, default=10)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--data-dir', type=str,
                      default='/media/chen/0B0D123F0B0D123F/Documents/research/kwai-extend/release/Files/data')
  parser.add_argument('--out-dir', type=str,
                      default='/media/chen/0B0D123F0B0D123F/Documents/research/kwai-extend/release/Files/output')
  parser.add_argument('--n-users', type=int, default=10986, help='number of users')
  parser.add_argument('--n-items', type=int, default=984983, help='number of items')
  parser.add_argument('--n-cates', type=int, default=512, help='number of categories')


  parameters = parser.parse_args(args)
  return parameters

def init_logging(args):
  log_name = os.path.join(args.out_dir, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
    args.phase, args.batch_size, args.lr, args.optimizer, args.w_reg, args.emb_reg, args.dropout_keep,
    args.neg_ratio, args.emb_dim, args.interaction, '-'.join([str(i) for i in args.agg_layers]),
    time.strftime('%Y%m%d-%H%M')))
  #log_name = os.path.join(params.out_dir, '{}.log'.format(time.strftime('%Y%m%d-%H%M')))

  logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename=log_name,
  )
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

  logging.info('phase: {}'.format(args.phase))
  logging.info('batch_size: {}'.format(args.batch_size))
  logging.info('max_epoch: {}'.format(args.max_epoch))
  logging.info('initial_learning_rate: {}'.format(args.lr))
  logging.info('optimizer: {}'.format(args.optimizer))
  logging.info('w_reg: {:.8f}'.format(args.w_reg))
  logging.info('emb_reg: {:.8f}'.format(args.emb_reg))
  logging.info('dropout_keep: {:.2f}'.format(args.dropout_keep))
  logging.info('neg_ratio: {}'.format(args.neg_ratio))
  logging.info('emb_dim: {}'.format(args.emb_dim))
  logging.info('interaction: {}'.format(args.interaction))
  logging.info('agg_layers: {}'.format('-'.join([str(i) for i in args.agg_layers])))
  logging.info('seed: {}'.format(args.seed))
  logging.info('display: {}'.format(args.display))
  logging.info('data_dir: {}'.format(args.data_dir))
  logging.info('out_dir: {}'.format(args.out_dir))
  logging.info('--------------------')


def main(args):
  args = process_args(args)
  directory = '{}'.format("CIN_Video")

  args.out_dir = os.path.join(args.out_dir, directory)
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

  init_logging(args)
  tf.set_random_seed(args.seed)

  # create model
  model = Model(args)

  # start running
  solver = Solver(model, args)
  if args.phase == 'train':
    solver.train()
  else:
    solver.test()


if __name__ == '__main__':
    main(sys.argv[1:])



























