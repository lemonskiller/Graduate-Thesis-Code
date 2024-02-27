# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pandas as pd

import random
from gcn.utils import *
from gcn.models import MLP, GCN
import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])  # 所有样本中有idx_train处置True，其余置False
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]  # 训练集部分用了实际label，其他用[0,0,0]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask  # 前三个都是871*3，后三个都是871


def run_training(algo, with_withouti, adj, weight_adj, features, labels, idx_train, idx_val, idx_test,
                 params):

    # Set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    # tf.set_random_seed(params['seed'])
    tf.set_random_seed(params['seed'])

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')

    # flags.DEFINE_bool('adaptive',params['adaptive'], 'whether to use adaptive')

    # Create test, val and train masked variables 创建masked变量（维度都变成871，方便后面计算）
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)

    # Some preprocessing 预处理，主要是选择模型
    features = preprocess_features(features)

    weight_support = chebyshev_polynomials(weight_adj, FLAGS.max_degree)

    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        # weight_support = [preprocess_adj(weight_adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby': ###用这个
        support = chebyshev_polynomials(adj, FLAGS.max_degree)  # 打印Calculating Chebyshev polynomials up to order 3...
        # weight_support = chebyshev_polynomials(weight_adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        # weight_support = [preprocess_adj(weight_adj)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')
    
    # Define placeholders 定义占位符？？？
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'weight_support': [tf.sparse_placeholder(tf.float32) for _ in range(len(weight_support))]
    }
    
    # Create model 创建模型 model_func=GCN
    model = model_func(placeholders, input_dim=features[2][1], num_nodes = adj.shape[0], adaptive = params['adaptive'],logging=True)  # depth=FLAGS.depth,


    # Initialize session
    sess = tf.Session()

    # Define model evaluation function 评估函数 *************************
    def evaluate(algo, with_withouti, feats, graph, weight_graph, label, mask, placeholder):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph, weight_graph, label, mask, placeholder)
        feed_dict_val.update({placeholder['phase_train'].name: False})
        outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
        #
        # Compute the area under curve 计算AUC
        pred = outs_val[2]
        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]  # 预测概率
        pred_label = np.argmax(pred, axis=1)  # 预测标签012
        lab_o = label
        lab_o = lab_o[np.squeeze(np.argwhere(mask == 1)), :]  # 实际标签onehot
        lab = np.argmax(lab_o, axis=1)  # 实际标签012

        recall = sklearn.metrics.recall_score(np.squeeze(lab), np.squeeze(pred_label), average='macro')
        pre = sklearn.metrics.precision_score(np.squeeze(lab), np.squeeze(pred_label), average='macro')
        f1 = sklearn.metrics.f1_score(np.squeeze(lab), np.squeeze(pred_label), average='macro')
        obj = sklearn.metrics.confusion_matrix(np.squeeze(lab), np.squeeze(pred_label),)
        auc = sklearn.metrics.roc_auc_score(np.squeeze(lab_o), np.squeeze(pred))

        # --我加--pred_label 预测标签012; lab 实际标签012
        test_label, test_pre = lab, pred_label

        # auc
        labels = np.unique(lab)
        if np.max(test_pre)==np.min(test_pre):
            test_auc = 0
        else:
            test_auc = roc_auc_score(test_label, label_binarize(test_pre, classes=labels), average='macro',
                                 multi_class='ovr')

        # 获取分类报告
        cr = classification_report(test_label, test_pre, output_dict=True)
        cr['auc'] = test_auc
        pd.DataFrame(cr).to_csv('./my_result/' + algo + '_' + with_withouti + '_meth.csv')

        ############### 混淆矩阵 #####################
        sns.set(font='Times New Roman', font_scale=1, style="ticks")  # ticks white

        fig = plt.figure()
        ax = fig.add_subplot(111)

        conf_mat = confusion_matrix(test_label, test_pre)

        if with_withouti == 'with':
            cmapi = 'summer'
        else:
            cmapi = 'Blues'

        fs=18
        h = sns.heatmap(conf_mat, annot=True, xticklabels=['NC', 'MCI', 'AD'],
                        yticklabels=['NC', 'MCI', 'AD'], fmt='g', cmap=cmapi,
                        annot_kws={'size': 26, 'weight': 'bold', 'family': 'Times New Roman'}, linewidths=1, cbar=True,
                        square=True)  # 画热力图

        # sns.heatmap(conf_mat/sum(sum(conf_mat)),annot=True,xticklabels=['NC','MCI','AD'],yticklabels=['NC','MCI','AD'],cmap='Blues',annot_kws={'size':20,'weight':'bold','family':'Times New Roman'},linewidths=0,cbar=False,square=True) #画热力图

        ax.set_title('Confusion Matrix', fontsize=fs + 2, weight='bold', family='Times New Roman')  # 标题
        ax.set_xlabel('Predict Label', fontsize=fs, weight='bold', family='Times New Roman')  # x轴
        ax.set_ylabel('True Label', fontsize=fs, weight='bold', family='Times New Roman')  # y轴

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs, weight='bold', family='Times New Roman')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs, weight='bold', family='Times New Roman', rotation=0)

        ax.grid(True, which='minor', linestyle='-', color='b')
        # plt.show()
        #
        # fig.savefig('./my_result/' + algo + '_heatmap_' + with_withouti + '_meth.svg', dpi=600, bbox_inches='tight')
        # plt.close()
        return outs_val[0], outs_val[1], recall, pre, f1, obj, auc,  (time.time() - t_test)
    
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    acc_val = []
    
    # Train model 训练模型
    for epoch in range(1, params['epochs']):
    
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, weight_support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout, placeholders['phase_train']: True})
    
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
        pred = outs[3]  # (1446,3)预测每种类别的概率值
        pred = pred[np.squeeze(np.argwhere(train_mask == 1)), :]  # 提取训练集(1301,3)
        labs = y_train  # (1446,3)实际标签onehot，其中测试集部分是[0,0,0]，训练集部分是实际值
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]  # 提取训练集(1301,3)
        train_auc = sklearn.metrics.roc_auc_score(np.squeeze(labs), np.squeeze(pred), average='macro')
    
        # Validation
        cost, acc, recall, pre, f1, obj, auc, duration = evaluate(algo, with_withouti,features, support, weight_support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        acc_val.append(acc)

        # Print results
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(outs[1]),
              "| train_acc=", "{:.5f}".format(outs[2]), "train_auc=", "{:.5f}".format(train_auc), "val_loss=", "{:.5f}".format(cost),
              "| val_acc=", "{:.5f}".format(acc), "val_auc=", "{:.5f}".format(auc), "f1", "{:.5f}".format(f1), 'pre',"{:.5f}".format(pre),'recall',"{:.5f}".format(recall))
    
        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):

        if epoch > FLAGS.early_stopping and acc_val[-1] > np.max(acc_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    
    print("Optimization Finished!")  # 训练结束
    
    # Testing 测试
    sess.run(tf.local_variables_initializer())
    test_cost, test_acc, test_recall, test_pre, test_f1, test_obj, test_auc, test_duration \
        = evaluate(algo, with_withouti,features, support, weight_support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "auc=", "{:.5f}".format(test_auc),
          "f1", "{:.5f}".format(test_f1),
          'pre', "{:.5f}".format(test_pre),
          'recall', "{:.5f}".format(test_recall)
          )
    
    return test_acc, test_recall, test_pre, test_f1, test_obj, test_auc
