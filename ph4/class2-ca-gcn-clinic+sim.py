#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import random

from scipy import sparse
from joblib import Parallel, delayed
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import label_binarize, normalize, MinMaxScaler

import train_GCN as Train
from Logger import make_print_to_file

import warnings
# import ipdb
warnings.filterwarnings("ignore")


def create_graph(scores, feature_pheno, num_nodes):
    '''
    不考虑对角线
    '''

    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = feature_pheno[l]
        if l in ['AGE', 'PTEDUCAT']:  # 连续数值变量
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict.values[k]) - float(label_dict.values[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:  # 离散数值变量，比如性别、APOE4
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict.values[k] == label_dict.values[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph

def create_weight_graph(scores, feature_pheno, num_nodes):
    '''
    不考虑对角线
    '''

    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = feature_pheno[l]
        if l in ['AGE', 'PTEDUCAT']:  # 连续数值变量
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict.values[k]) - float(label_dict.values[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:  # 离散数值变量，比如性别、APOE4
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict.values[k] == label_dict.values[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph

def my_train_test_split(label, fold, test_per_class, val_per_class):
    '''
    输入：label--标签,fold--交叉验证折数
    输出：tuple类型，fold个成员，每个成员为每个fold的train_ind,test_ind,val_ind
    '''
    train_test_val_ind={}
    # 划分 val, test, trian: 随机等比例采样得 valid，test；bootstrap 采样得训练集

    label1_ind = np.where(label==1)
    label2_ind = np.where(label==2)

    test_ind1=label1_ind[0][:test_per_class]
    test_ind2=label2_ind[0][:test_per_class]

    val_ind1=label1_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind2=label2_ind[0][test_per_class:test_per_class+val_per_class]

    remain_train_ind1 = label1_ind[0][test_per_class :]
    remain_train_ind2 = label2_ind[0][test_per_class :]

    # test val index
    test_ind=list(test_ind1)+list(test_ind2)
    val_ind =list(val_ind1)+list(val_ind2)

    # remain train
    train_per_class=min([len(remain_train_ind1),len(remain_train_ind2)])
    num_ind1 = len(remain_train_ind1)
    num_ind2 = len(remain_train_ind2)

    for i in range(fold):
        random.seed(i)
        # 从中随机抽取train_per_class个

        train_ind1 = random.sample(list(remain_train_ind1), train_per_class)
        train_ind2 = random.sample(list(remain_train_ind2), train_per_class)
        train_ind=list(train_ind1)+list(train_ind2)

        train_test_val_ind[i]=(train_ind,test_ind,val_ind)
    return tuple(train_test_val_ind.values())


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(algo, with_withouti,train_ind, test_ind, val_ind, graph_feat, weight_graph_feat, features, features_mri,features_meth,features_grade, y, y_data, params):
    """
        train_ind       : indices of the training samples 训练集序号
        test_ind        : indices of the test samples 测试集序号
        val_ind         : indices of the validation samples 验证集序号
        graph_feat      : population graph computed from phenotypic measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features 特征向量
        y               : ground truth labels (num_subjects x 1) 标签
        y_data          : ground truth labels - different representation (num_subjects x 2) 把1/2变成[0,1]/[1,0]的标签
        params          : dictionnary of GCNs parameters GCN的参数

    returns:
        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the linear classifier
        lin_auc     : average area under curve over the test samples using the linear classifier
        fold_size   : number of test samples
    """

    ''' 横向拼接
        np.hstack((n1, n2))'''

    # 分类用特征
    # x_data = np.hstack((features_grade, features_mri))
    x_data = features
    distv = distance.pdist(x_data, metric='correlation')  # x_data:样本*特征
    dist = distance.squareform(distv)  # n个样本，两两配对，共 n*(n-1)/2
    sigma = np.mean(dist)  # dist中所有元素的均值
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))  # 相似性度量公式

    # feature selection/dimensionality reduction step 特征选择/降维
    x_data1 = features_mri
    distv1 = distance.pdist(x_data1, metric='correlation')  # x_data:样本*特征
    dist1 = distance.squareform(distv1)  # n个样本，两两配对，共 n*(n-1)/2
    sigma1 = np.mean(dist1)  # dist中所有元素的均值
    sparse_graph_mri = np.exp(- dist1 ** 2 / (2 * sigma1 ** 2))  # 相似性度量公式


    x_data2 = features_meth
    distv2 = distance.pdist(x_data2, metric='correlation')  # x_data:样本*特征
    dist2 = distance.squareform(distv2)  # n个样本，两两配对，共 n*(n-1)/2
    sigma2 = np.mean(dist2)  # dist中所有元素的均值
    sparse_graph_meth = np.exp(- dist2 ** 2 / (2 * sigma2 ** 2))  # 相似性度量公式


    x_data3 = features_grade
    distv3 = distance.pdist(x_data3, metric='correlation')  # x_data:样本*特征
    dist3 = distance.squareform(distv3)  # n个样本，两两配对，共 n*(n-1)/2
    sigma3 = np.mean(dist3)  # dist中所有元素的均值
    sparse_graph_grade = np.exp(- dist3 ** 2 / (2 * sigma3 ** 2))  # 相似性度量公式

    if params['priorInfo']:
        final_graph = graph_feat * sparse_graph  # 临床*[MRI,meth,grade] 对应元素相乘；acc 82%-88%
    else:
        final_graph = sparse_graph

    #fixme
    final_weight_graph = weight_graph_feat*sparse_graph

    # --------- 无视！！---------
    # 用线性分类器分类
    clf = RidgeClassifier()  # 岭分类器
    # clf = LogisticRegression()
    clf.fit(features[train_ind, :], y[train_ind].ravel())
    y_pred = clf.predict(features[test_ind, :])  # 形式一：原始值（0或1或2）
    # y_pro = clf.predict_proba(features[test_ind, :])  # 形式二：各类概率值
    # y_onehot = label_binarize(y_pred, np.arange(y_data.shape(1)))  # 形式三：one-hot值
    # Compute the accuracy 计算准确度acc/召回率recall/精度precision/F-score/auc
    lin_acc = metrics.accuracy_score(y[test_ind], y_pred)
    lin_recall = metrics.recall_score(y[test_ind], y_pred, average='macro')
    lin_pre = metrics.precision_score(y[test_ind], y_pred, average='macro')
    lin_f1 = metrics.f1_score(y[test_ind], y_pred, average='macro')
    lin_obj = metrics.confusion_matrix(y[test_ind].ravel(), y_pred)
    print('Linear accuracy:\t{}'.format(lin_acc))
    # --------- 无视！！---------

    # Classification with GCNs 用GCN分类
    test_acc, test_recall, test_pre, test_f1, test_obj, test_auc = Train.run_training(algo, with_withouti,
        final_graph, final_weight_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind, test_ind, params)  # train_GCN.py

    return lin_acc, lin_recall, lin_pre, lin_f1, lin_obj, \
           test_acc, test_recall, test_pre, test_f1, test_obj, test_auc


def main():
    # algo = 'graph_age_sex_mri_feat_meth_grade' # 85.238
    # algo = 'graph_age_sex_meth_feat_mri_grade' # 81.429
    algo = 'class2-cagcn-clinic-sim'

    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the TADPOLE dataset')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')  # 0.02
    parser.add_argument('--decay', default=1e-5, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4, 1e-5)')
    parser.add_argument('--hidden', default=5, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.05, type=float, help='Initial learning rate (default: 0.005,0.01)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    parser.add_argument('--epochs', default=92, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')

    parser.add_argument('--seed', default=2020, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=5, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=0, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    parser.add_argument('--early_stopping', default=100, help='early stop')

    # fixme modify the augments below to construct different models
    parser.add_argument('--with_meth', default=True, type=bool, help='whether to use meth data')

    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                       'uses chebyshev polynomials, '
                                                       'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--adaptive', default=True, type=bool, help='whether to use adaptive adj')
    parser.add_argument('--priorInfo', default = True, type = bool, help = 'whether to use prior information')

    args = parser.parse_args()

    # GCN Parameters
    params = dict()
    params['model'] = args.model  # gcn model using chebyshev polynomials gcn模型用切比雪夫多项式
    params['lrate'] = args.lrate  # Initial learning rate 初始学习率
    params['epochs'] = args.epochs  # Number of epochs to train 训练的epochs数量
    params['dropout'] = args.dropout  # Dropout rate (1 - keep probability) dropout比率
    params['hidden'] = args.hidden  # Number of units in hidden layers 隐藏层单元数
    params['decay'] = args.decay  # Weight for L2 loss on embedding matrix. 嵌入矩阵中L2损失的权重
    params['early_stopping'] = params['epochs']  # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3  # Maximum Chebyshev polynomial degree. 切比雪夫多项式最大维度
    params['depth'] = args.depth  # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed  # seed for random initialisation 随机种子初始化
    params['adaptive'] = args.adaptive
    params['priorInfo'] = args.priorInfo
    params['num_features'] = args.num_features  # number of features for feature selection step 特征选择步骤的特征数量
    params['num_training'] = args.num_training  # percentage of training set used for training 训练集的占比
    atlas = args.atlas  # atlas for network construction (node definition) 网络结构的atlas（节点定义）
    connectivity = args.connectivity  # type of connectivity used for network construction 网络结构的连接类型

    make_print_to_file(path='./log/')  # 自己加的，打印部分生成log文件
    print(str(params))
    # 导入数据
    # ######## 自己加的，获取数据 ###############
    # Load data

# 2分类：
    df = pd.read_csv('TADPOLE_D1_D2_del18_meth_1397_mci2.csv')

    if args.with_meth:
        with_withouti = 'with'
    else:
        with_withouti = 'without'
        params['epochs'] = 251
    print('* '*6,with_withouti,'* '*6)
    # 性别
    gender_ind = 5
    # 年龄
    age_ind = 4
    # APOE4
    apoe_ind = 6
    # label
    label_ind = 17
    # 评分 ADAS11,ADAS13,MMSE,CDRSB,RAVLT
    grade_ind = np.arange(8, 17)
    # 受教育年限
    edu_ind = 7
    # basic
    basic_ind = [age_ind, gender_ind, edu_ind, apoe_ind]
    # MRI
    mri_ind = np.arange(19, 37)
    # 甲基化
    meth_ind = np.arange(37, 52)

    labels = np.floor(df['label_NC0_MCI1_AD2'].values.astype(float)).astype(int)  # str转为浮点数，再转为int

# 2分类
    num_classes = 2
    
    num_nodes = len(labels)  # 1397

    labels_onehot = label_binarize(labels, np.arange(num_classes)).reshape(num_nodes)

    labels_onehot = np.eye(num_classes)[labels_onehot]

    print('classes:\t', str(num_classes))
    print('subjects:\t', str(num_nodes))

    if with_withouti=='with':
        feat_ind = list(basic_ind)+list(grade_ind)+list(mri_ind)+list(meth_ind) # list(basic_ind)+
    else:
        feat_ind = list(basic_ind)+list(grade_ind)+list(mri_ind)  #  list(basic_ind)+

    # NORMALIZE
    features = df.iloc[:,feat_ind].values.astype(float)  # 所有特征构图 82
    features_mri = df.iloc[:, mri_ind].values.astype(float)
    features_meth = df.iloc[:, meth_ind].values.astype(float)
    features_grade = df.iloc[:, grade_ind].values.astype(float)

    print('feature num:\t', len(features[0]))
    features = MinMaxScaler().fit_transform(features.astype(np.float))
    features_mri = MinMaxScaler().fit_transform(features_mri.astype(np.float))
    features_meth = MinMaxScaler().fit_transform(features_meth.astype(np.float))
    features_grade = MinMaxScaler().fit_transform(features_grade.astype(np.float))

    # 用表型特征计算人群图权重
    cognitive = ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
                 'RAVLT_perc_forgetting', 'FAQ']
    factors = ['AGE', 'gender_male0_female1', 'PTEDUCAT', 'APOE4'] 
    graph_use = factors  # ###########
    print('risk factor: \t' + str(len(graph_use)) + '\n' + str(graph_use))
    graph = create_graph(graph_use, df[graph_use], num_nodes)  # 1397*1397

    weight_graph = create_weight_graph(graph_use, df[graph_use], num_nodes)

    if args.folds == 5:  # run cross validation on all folds
        # def train_fold(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs):
        #   return
        #         test_acc    : average accuracy over the test samples using GCNs
        #         test_auc    : average area under curve over the test samples using GCNs
        #         lin_acc     : average accuracy over the test samples using the linear classifier
        #         lin_auc     : average area under curve over the test samples using the linear classifier
        #         fold_size   : number of test samples
        scores = Parallel(n_jobs=1)(delayed(train_fold)(algo, with_withouti,train_ind, test_ind, test_ind, graph, weight_graph, features, features_mri,features_meth,features_grade,
                                                         labels, labels_onehot, params)
                                     for train_ind, test_ind ,val_ind in
                                     list(my_train_test_split(labels,5,30,10)))
                                     # reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(labels)))))
        # {list:10}，每个{tuple:2}，一个1301，一个145，由于folds默认是0，所以只取第一个list

        print('******************** 5 folds finished********************')
        # lin_acc, lin_recall, lin_pre, lin_f1, lin_obj, test_acc, test_recall, test_pre, test_f1, test_obj
        scores_lin_acc = [x[0] for x in scores]
        scores_lin_recall = [x[1] for x in scores]
        scores_lin_pre = [x[2] for x in scores]
        scores_lin_f1 = [x[3] for x in scores]
        scores_lin_obj = [x[4] for x in scores]

        scores_acc = [x[5] for x in scores]
        scores_recall = [x[6] for x in scores]
        scores_pre = [x[7] for x in scores]
        scores_f1 = [x[8] for x in scores]
        scores_obj = [x[9] for x in scores]
        scores_auc = [x[10] for x in scores]

        print('lin_fold\tacc\trecall\tprecision\tf1\tconfusion_matrix')
        for i in range(5):
            print('%d\t%.4f\t%.4f\t%.4f\t%.4f' %
                  (i+1, scores_lin_acc[i], scores_lin_recall[i], scores_lin_pre[i], scores_lin_f1[i]))
            print(scores_lin_obj[i].T)
        print('GCN_fold\tacc\trecall\tprecision\tf1\tauc\tconfusion_matrix')
        for i in range(5):
            print('%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
                  (i+1, scores_acc[i], scores_recall[i], scores_pre[i], scores_f1[i], scores_auc[i]))
            print(scores_obj[i].T)

        print('******************** 5-fold average results********************')
        print('overall\tacc\trecall\tprecision\tf1\tauc')
        print('linear\t%.4f\t%.4f\t%.4f\t%.4f'
              % (np.mean(scores_lin_acc), np.mean(scores_lin_recall), np.mean(scores_lin_pre), np.mean(scores_lin_f1)))
        print('GCN\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'
              % (np.mean(scores_acc), np.mean(scores_recall), np.mean(scores_pre), np.mean(scores_f1), np.mean(scores_auc)))
        print('linear average confusion matrix:\n', np.mean(scores_lin_obj, axis=0).T)
        print('GCN average confusion matrix:\n', np.mean(scores_obj, axis=0).T)

    else:  # compute results for only one fold #fold默认值是0
        # 分层采样划分数据集，交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_splits = list(skf.split(features, np.squeeze(labels)))  # {list:10}，每个{tuple:2}

        train = cv_splits[args.folds][0]  # 训练集
        test = cv_splits[args.folds][1]  # 测试集
        val = test

        scores_lin_acc, scores_lin_recall, scores_lin_pre, scores_lin_f1, scores_lin_obj, \
            scores_acc, scores_recall, scores_pre, scores_f1, scores_obj, scores_auc \
            = train_fold(train, test, val, graph, weight_graph,features, labels, labels_onehot, params)

        print('overall\tacc\trecall\tprecision\tf1\tauc')
        print('linear\t%.4f\t%.4f\t%.4f\t%.4f'
              % (np.mean(scores_lin_acc), np.mean(scores_lin_recall), np.mean(scores_lin_pre), np.mean(scores_lin_f1)))
        print('GCN\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'
              % (np.mean(scores_acc), np.mean(scores_recall), np.mean(scores_pre), np.mean(scores_f1), np.mean(scores_auc)))
        print('linear average confusion matrix:\n', scores_lin_obj.T)
        print('GCN average confusion matrix:\n', scores_obj.T)


if __name__ == "__main__":
    import tensorflow as tf

    FLAGS = tf.app.flags.FLAGS
    lst = list(FLAGS._flags().keys())
    for key in lst:
        print('key: ',key)
        FLAGS.__delattr__(key)

    main()
