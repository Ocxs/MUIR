#_*_coding:utf-8_*_
#author: Xusong Chen
#date: 3/17/19 10:49 PM

import numpy as np
import os
from tqdm import tqdm


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

def sigmoid(x):
  return 1./(1+np.exp(-x))

def compute_fusion_result():
  pass

def main():
  data_dir = '/home/chen/Music/Video'
  if 1:
    dataset = 'Video'
  else:
    dataset = 'Movie'

  #for emb_dim in [64, 32, 16, 8]:
  emb_dim = 64
  #for hist_length in [20, 40, 80, 120, 160, 200, 240]:
  for hist_length in [160]:
    print('emb_dim: {}, hist_length: {}'.format(emb_dim, hist_length))
    lin_pred = np.load(os.path.join(data_dir, 'LIN_{}_pred_{}.npy'.format(dataset, emb_dim))).item()
    nin_pred = np.load(os.path.join(data_dir, 'NIN_{}_pred_{}.npy'.format(dataset, emb_dim))).item()
    cin_pred = np.load(os.path.join(data_dir, 'CIN_{}_pred_{}.npy'.format(dataset, emb_dim))).item()
    iin_pred = np.load(os.path.join(data_dir, 'IIN_{}_pred_{}_{}.npy'.format(dataset, emb_dim, hist_length))).item()

    # lin_pred = np.load(os.path.join(data_dir, 'LIN_{}_pred.npy'.format(dataset))).item()
    # nin_pred = np.load(os.path.join(data_dir, 'NIN_{}_pred.npy'.format(dataset))).item()
    # cin_pred = np.load(os.path.join(data_dir, 'CIN_{}_pred.npy'.format(dataset))).item()
    # iin_pred = np.load(os.path.join(data_dir, 'IIN_{}_pred.npy'.format(dataset))).item()

    print('load data over')

    fusion_types = ['MUIR', 'MUIR-LN', 'MUIR-CI', 'MUIR-L', 'MUIR-N', 'MUIR-C', 'MUIR-I', 'L', 'N', 'C', 'I']
    fusion_types = ['MUIR-LI', 'MUIR-LN', 'MUIR-LC', 'MUIR-IN', 'MUIR-IC', 'MUIR-NC']
    for type in fusion_types:
      print(type, end=' ')
      fusion_pred = {}
      for uid in nin_pred:
        items = nin_pred[uid]
        length = len(items)

        if type == 'MUIR':
          fusion_pred[uid] = [[lin_pred[uid][i][0] + nin_pred[uid][i][0] + iin_pred[uid][i][0] + cin_pred[uid][i][0],  # + iin_pred[uid][i][0]+nin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-L':
          fusion_pred[uid] = [[nin_pred[uid][i][0] + iin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-N':
          fusion_pred[uid] = [[lin_pred[uid][i][0] + iin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-I':
          fusion_pred[uid] = [[nin_pred[uid][i][0] + lin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-C':
          fusion_pred[uid] = [[nin_pred[uid][i][0] + iin_pred[uid][i][0] + lin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-LI':
          fusion_pred[uid] = [[nin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-LN':
          fusion_pred[uid] = [[cin_pred[uid][i][0] + iin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-LC':
          fusion_pred[uid] = [[nin_pred[uid][i][0] + iin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-IN':
          fusion_pred[uid] = [[lin_pred[uid][i][0] + cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-IC':
          fusion_pred[uid] = [[lin_pred[uid][i][0] + nin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'MUIR-NC':
          fusion_pred[uid] = [[lin_pred[uid][i][0] + iin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]

        elif type == 'L':
          fusion_pred[uid] = [[lin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'N':
          fusion_pred[uid] = [[nin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'C':
          fusion_pred[uid] = [[cin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]
        elif type == 'I':
          fusion_pred[uid] = [[iin_pred[uid][i][0],
                               nin_pred[uid][i][1], nin_pred[uid][i][2]]
                              for i in range(length)]

      auc = compute_auc(fusion_pred)
      print('auc:{:.4f}'.format(auc), end=' ')

      precisions, recalls, ndcgs = [], [], []
      for k in [1, 2, 5]:
        top_k = k * 10
        precision, recall, ndcg = compute_p_and_r_and_ndcg(fusion_pred, top_k)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

      print('precision:{}'.format(','.join(['{:.4f}'.format(p) for p in precisions])), end=' ')
      print('recall:{}'.format(','.join(['{:.4f}'.format(p) for p in recalls])), end=' ')
      print('ndcg:{}'.format(','.join(['{:.4f}'.format(ndcg) for ndcg in ndcgs])), end=' ')
      print()


if __name__ == '__main__':
  main()