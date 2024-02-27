# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import argparse
import random
import numpy as np
import pandas as pd
# import scipy.sparse as sp
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, auc
import matplotlib.pyplot as plt


from sklearn.utils import shuffle

import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import  MinMaxScaler, label_binarize
from utils import accuracy, sparse_mx_to_torch_sparse_tensor, normalize
from models_dense import GCN

'''
想用年龄,性别,受教育程度，APOE构建邻接矩阵，MRI+meth特征拼接作为每个节点的特征向量
'''

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, full_adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, full_adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_test], labels[idx_test]) #上测试集
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        ' | acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        ' | acc_val: {:.4f}'.format(acc_val.item()),
        'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj, full_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))

    ######### 以下都是自己加的，画混淆矩阵和分类报告文件 #####################
    test_label,test_pre_tmp= labels[idx_test],output[idx_test]
    test_pre = test_pre_tmp.max(1)[1].type_as(test_label)
    test_label = test_label.cpu().numpy()
    test_pre = test_pre.cpu().detach().numpy()

    print('test pre: ')
    print(test_pre)

    # auc
    test_auc = roc_auc_score(test_label, label_binarize(test_pre, classes=np.unique(label)),average='macro',multi_class = 'ovr')
    # 获取分类报告
    cr = classification_report(test_label, test_pre, output_dict=True)
    cr['auc']=test_auc
    pd.DataFrame(cr).to_csv('../adj_eye/cgcn_'+with_withouti+'_meth_mci.csv')
    
    sns.set(font='Times New Roman',font_scale=1,style="ticks")  # ticks white

    fig = plt.figure()
    ax = fig.add_subplot(111)

    conf_mat = confusion_matrix(test_label,test_pre)

    if with_withouti=='with':
        cmapi='summer'
    else:
        cmapi='Blues'

    h = sns.heatmap(conf_mat,annot=True,xticklabels=['MCI-MCI','MCI-AD'],yticklabels=['MCI-MCI','MCI-AD'], fmt='g',cmap=cmapi,annot_kws={'size':26,'weight':'bold','family':'Times New Roman'},linewidths=1,cbar=True,square=True) #画热力图

    ax.set_title('Confusion Matrix',fontsize=fs+2,weight='bold',family='Times New Roman') #标题
    ax.set_xlabel('Predict Label',fontsize=fs,weight='bold',family='Times New Roman') #x轴
    ax.set_ylabel('True Label',fontsize=fs,weight='bold',family='Times New Roman') #y轴

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs,weight='bold',family='Times New Roman')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs,weight='bold',family='Times New Roman', rotation=0)

    ax.grid(True, which='minor', linestyle='-',color='b')
    plt.show()

    fig.savefig('../adj_eye/cgcn_'+with_withouti+'_heatmap_mci.svg', dpi=600, bbox_inches='tight')

# ######## 自己加的 ###############
def create_graph(scores, feature_pheno, num_nodes):
    graph = np.zeros((num_nodes, num_nodes))
    for l in scores:
        label_dict = feature_pheno[l]
        if l in ['AGE', 'PTEDUCAT']:  # 数值连续的特征，年龄，受教育时长
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict.values[k]) - float(label_dict.values[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:  # 数值离散的特征，比如性别（0/1）、APOE4（0/1）、
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict.values[k] == label_dict.values[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph
# ######## 自己加的 ###############

# --------- 为可复现加的 -----------
fs=18 # 画图字体
seed = 2020
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# --------- 为可复现加的 -----------

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=seed, help='Random seed.')
parser.add_argument('--epochs', type=int, default=13,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,#0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,#16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,#0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ######## 自己加的，获取数据 ###############
# Load data
df = pd.read_csv('TADPOLE_D1_D2_del18_meth_1397_mci2.csv')
shuffle(df,random_state=2020,replace=True)

# 性别
gender_ind=5
# 年龄
age_ind=4
# APOE4
apoe_ind=6
# label
label_ind=17
# 评分 ADAS11,ADAS13,MMSE,CDRSB,RAVLT
grade_ind=np.arange(8,17)
# 受教育年限
edu_ind=7
# basic
basic_ind = [age_ind,gender_ind,edu_ind,apoe_ind]
# MRI
mri_ind=np.arange(19,37)
# 甲基化
meth_ind=np.arange(37,52)
# 加不加甲基化特征
# with_without=['with']
with_without=['without']

# 用表型特征构图
cognitive = ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
                'RAVLT_perc_forgetting', 'FAQ']
factors = ['AGE', 'gender_male0_female1', 'PTEDUCAT', 'APOE4']
graph_use = factors 

print('risk factor: \t' + str(len(graph_use)) + '\n' + str(graph_use))
graph = create_graph(graph_use, df[graph_use], df.shape[0])  # 1397*1397
# graph = graph/graph.max() # 归一化

for with_withouti in with_without:
    print('-'*6)
    print(with_withouti,' Methylation')
    if with_withouti=='with':
        feat_ind = list(grade_ind)+list(mri_ind)+list(meth_ind)
    if with_withouti=='without':
        feat_ind = list(grade_ind)+list(mri_ind)  

    # NORMALIZE
    sample_feat = df.iloc[:,feat_ind].values
    sample_feat = MinMaxScaler().fit_transform(sample_feat.astype(np.float)) # 82

    label = df.iloc[:,label_ind].values

    # 划分 val, test, trian: 随机等比例采样得 valid，test；bootstrap 采样得训练集
    # label0_ind = np.where(label==0)
    label1_ind = np.where(label==1)
    label2_ind = np.where(label==2)

    test_per_class=30 # 测试集共60个，每个类别30个
    val_per_class=10 # 验证集共20个，每个类别10个
    # test
    # test_ind0=label0_ind[0][:test_per_class]
    test_ind1=label1_ind[0][:test_per_class]
    test_ind2=label2_ind[0][:test_per_class]
    # val
    # val_ind0=label0_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind1=label1_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind2=label2_ind[0][test_per_class:test_per_class+val_per_class]
    # remain train
    # remain_train_ind0=label0_ind[0][test_per_class+val_per_class:]
    remain_train_ind1=label1_ind[0][test_per_class+val_per_class:]
    remain_train_ind2=label2_ind[0][test_per_class+val_per_class:]

    # test val index
    test_ind=list(test_ind1)+list(test_ind2)
    val_ind=list(val_ind1)+list(val_ind2)
    
    train_per_class=min([len(remain_train_ind1),len(remain_train_ind2)])

    # train_ind0=remain_train_ind0[:train_per_class]
    train_ind1=remain_train_ind1[:train_per_class]
    train_ind2=remain_train_ind2[:train_per_class]
    train_ind=list(train_ind1)+list(train_ind2)

    features = torch.FloatTensor(sample_feat)
    labels = torch.LongTensor(label)

    # 单位矩阵
    # adj = sp.coo_matrix(np.eye(sample_feat.shape[0]),
    #                 shape=(sample_feat.shape[0], sample_feat.shape[0]),
    #                 dtype=np.float32)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    
    # basic feat 矩阵
    adj = torch.from_numpy(np.eye(sample_feat.shape[0]))
    # adj = torch.from_numpy(graph)

    # 全连接矩阵
    # full_adj = sp.coo_matrix(np.ones([adj.shape[0], adj.shape[0]]),
    #                 shape=(adj.shape[0], adj.shape[0]),
    #                 dtype=np.float32)
    # full_adj = sparse_mx_to_torch_sparse_tensor(full_adj)
    full_adj = adj 

    idx_train = torch.LongTensor(np.array(train_ind))
    idx_val = torch.LongTensor(np.array(val_ind))
    idx_test = torch.LongTensor(np.array(test_ind))

    print('sample_feat: ',sample_feat.shape[0],' ',sample_feat.shape[1])
    print('MCI-MCI num: ',len(label1_ind[0]),'   MCI-AD num: ',len(label2_ind[0]))
    print('train num: ',len(idx_train))
    print('valid num: ',len(idx_val))
    print('test num: ',len(idx_test))

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        full_adj = full_adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()





