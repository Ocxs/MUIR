#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

def var_init(name, shape, initializer=tf.contrib.layers.xavier_initializer(),
              trainable=True):
  with tf.device('/gpu:0'):
    var = tf.get_variable(
      name=name,
      shape=shape,
      initializer=initializer,
      trainable=trainable
    )
    if not tf.get_variable_scope().reuse and name != 'train_cover_image_feature':
      tf.add_to_collection("parameters", var)
    if name == 'train_cover_image_feature':
      tf.add_to_collection('train_cover_image_feature', var)
    return var

def dense(x,
          units,
          name,
          keep_prob=1.0,
          activation= None,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          bias_initializer=tf.zeros_initializer(),
          reuse=None):
  """
  Functional interface for the densely-connected layer.
  :param x: Tensor input.
  :param units: Integer or Long, dimensionality of the output space.
  :param name: String, the name of the parameter.
  :param keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
  :param activation: Activation function (callable). Set it to None to maintain a linear activation.
  :param kernel_initializer:
  :param bias_initializer:
  :param reuse:
  :return:
  """
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    if not isinstance(name, (list, tuple)):
      raise ValueError('name should be list or tuple')

    prev_units= x.get_shape().as_list()[-1]
    w = var_init(name[0], (prev_units, units), kernel_initializer)
    out = tf.tensordot(x, w, axes=[[-1], [0]])
    if len(name) > 1:
      b = var_init(name[1], units, bias_initializer)
      out += b

    if activation is not None:
      out = activation(out)

    out = tf.nn.dropout(out, keep_prob)
    return out


def mlp(x, fusion_layers, keep_prob):
  _, dim = x.get_shape().as_list()
  n_layers = len(fusion_layers)
  for i in range(n_layers):
    x = dense(x, fusion_layers[i], ['w{}'.format(i+1), 'b{}'.format(i+1)], keep_prob, tf.nn.relu)
  return x

def dnn(x,
        fusion_layers,
        keep_prob):
  """
  Feedforward network.
  :param x: Tensor input.
  :param fusion_layers: List, the layers of feedforward network, like [256, 128].
  :param keep_prob:
  :return: The micro-video click-through probabilities.
  """
  n_layers = len(fusion_layers)
  x = mlp(x, fusion_layers, keep_prob)

  logit = dense(x, 1, ['w{}'.format(n_layers+1), 'b{}'.format(n_layers+1)])
  return tf.squeeze(logit)


def neighbor_attention(neighbor_user_emb,
                       item_emb,
                       avg_neighbor_emb,
                       keep_prob,
                       att_act,
                       scale):
  """
  perform attention on neighbor users
  :param neighbor_user_emb: [N, n_neighbors, user_dim]
  :param item_emb: [N, item_dim]
  :return: the representation of neighbor users: [N, user_dim]
  """
  if att_act == 'relu':
    act_func = tf.nn.relu
  elif att_act == 'sigmoid':
    act_func = tf.nn.sigmoid
  elif att_act == 'tanh':
    act_func = tf.nn.tanh
  else:
    raise ValueError('invalid att_act: {}'.format(att_act))


  n_shape = neighbor_user_emb.get_shape().as_list()
  i_proj = tf.tile(tf.expand_dims(item_emb, axis=1), [1, n_shape[1], 1])
  avg_proj = tf.tile(avg_neighbor_emb, [1, n_shape[1], 1])

  n_proj = dense(tf.concat([i_proj, neighbor_user_emb-avg_proj], axis=-1), n_shape[-1], ['neighbor_w', 'neighbor_b'], keep_prob)
  proj = act_func(n_proj)
  w = dense(proj, n_shape[-1], ['att_w', 'att_b'])
  if scale:
    w = w / (n_shape[-1] ** 0.5)
  alpha = tf.nn.softmax(w, dim=1)
  att_vec = tf.reduce_sum(alpha * neighbor_user_emb, axis=1)


  return att_vec

def item_interest_attention(x, x_mem, q, avg, mask, keep_prob, att_act, scale, att_dropout):
  if att_act == 'relu':
    act_func = tf.nn.relu
  elif att_act == 'sigmoid':
    act_func = tf.nn.sigmoid
  elif att_act == 'tanh':
    act_func = tf.nn.tanh
  else:
    raise ValueError('invalid att_act: {}'.format(att_act))

  x_shape = x.get_shape().as_list()
  q = tf.tile(tf.expand_dims(q, axis=1), [1, x_shape[1], 1])
  avg = tf.tile(tf.expand_dims(avg, axis=1), [1, x_shape[1], 1])
  x_proj = dense(tf.concat([x, q, avg], axis=-1), x_shape[-1], ['x_w1', 'x_b1'], keep_prob)

  proj = act_func(x_proj)
  w = dense(proj, x_shape[-1], ['w2'], keep_prob)
  mask_w = softmax_mask(w, mask)
  if scale:
    mask_w = mask_w / (x.get_shape().as_list()[-1] ** 0.5)
  alpha = tf.nn.softmax(mask_w, dim=len(x_shape)-2)
  alpha = tf.nn.dropout(alpha, keep_prob=att_dropout)
  att_vec = tf.reduce_sum(alpha * x_mem, axis=len(x_shape)-2)
  return att_vec


def softmax_mask(x, mask):
  """
  :param x: [n,m,d] or [n,b,m,d]
  :param mask: [n,m] or [n,b,m]
  :return:
  """
  x_shape = x.get_shape().as_list()
  pad_num = len(x_shape)-1
  mask = tf.tile(tf.expand_dims(mask, axis=-1), pad_num*[1]+[x_shape[-1]])
  paddings = tf.ones_like(mask, tf.float32) * (-2 ** 32 + 1)
  softmax_mask = tf.where(mask, x, paddings)
  return softmax_mask