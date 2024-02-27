# -*- coding:utf-8 -*-

# from openpyxl import load_workbook
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import matplotlib
import pandas as pd

import sys
libsvm_path='/home/lemon/sgld/libsvm-3.24/python'
sys.path.append(libsvm_path)

from svmutil import svm_train,svm_predict
from sklearn.model_selection import train_test_split
import scipy.stats as ss
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# libsvm基于交叉验证的网格搜参
def gridsearch_select(sample_feat,label,fold):

    C_group = range(-1,15,2)
    C_group = [2**x for x in C_group]

    gamma_group = range(7,-15,-2)
    gamma_group = [2**x for x in gamma_group]

    score = []
    for C in C_group:
        for gamma in gamma_group:
            svmopt='-v '+str(fold)+' -c '+str(C)+' -g '+str(gamma)
            print(svmopt)
            score.append(svm_train(sample_feat,label,svmopt))
    best_score = max(score)
    index = score.index(best_score)
    best_C = C_group[index//len(gamma_group)]
    best_g = gamma_group[index%len(gamma_group)]
    return best_C,best_g,best_score


#-----load files------------
fold=5
featnum=30
df=pd.read_csv(r'ROSMAP_497_GEO_488_del_nagene.csv')

print('--- loading files ---')
# test
test_label=df.iloc[0,2+236+261:].values.astype(float)
test_sample_feat=df.iloc[1:,2+236+261:].values.T.astype(float)

# train
train_label=df.iloc[0,2:2+236+261].values.astype(float)
train_sample_feat=df.iloc[1:,2:2+236+261].values.T.astype(float)

''' 交换训练集测试集 '''
test_sample_feat,train_sample_feat=train_sample_feat,test_sample_feat
test_label,train_label=train_label,test_label

train_sample_feat0=train_sample_feat[train_label!=1,:]
train_sample_feat1=train_sample_feat[train_label==1,:]

#------training------
# 训练集交叉验证，测试集测试
X = train_sample_feat
Y = train_label

#------training------
# 训练集交叉验证，测试集测试
percent=[1]#[0.5, 0.7, 0.9,1]
total_repeat = 1
for percenti in percent:
    res = pd.DataFrame()
    for repeati in range(1,1+total_repeat):

        X[X < 0] = 0
        X[X > 1] = 1
        test_sample_feat[test_sample_feat < 0] = 0
        test_sample_feat[test_sample_feat > 1] = 1
        # ------ select features by pca ----------------
        from sklearn.decomposition import PCA

        # --------- cv & test ----------
        cv_acc = []
        test_acc = []
        test_recall = []
        test_f1 = []
        test_precision = []
        test_auc = []
        for i in range(1, 1 + featnum):
            print('--- making feature ', str(i), '---')
            pca = PCA(n_components=i)
            select_x = pca.fit_transform(X)
            best_c, best_g, best_acc = gridsearch_select(Y, select_x, fold)
            best_svmopt = '-c ' + str(best_c) + ' -g ' + str(best_g)
            model = svm_train(Y, select_x, best_svmopt)
            # svm predict
            test_select_x = pca.fit_transform(test_sample_feat)

            pre_test_label, p_test_acc, p_val = svm_predict(test_label, test_select_x, model)
            test_recalli = recall_score(test_label, pre_test_label)
            test_f1i = f1_score(test_label, pre_test_label, average='binary')
            test_precisioni = precision_score(test_label, pre_test_label, average='binary')
            test_acci = p_test_acc[0]
            test_auci = roc_auc_score(test_label, pre_test_label)

            test_recall.append(test_recalli)
            test_f1.append(test_f1i)
            test_precision.append(test_precisioni)

            test_acc.append(test_acci)
            test_auc.append(test_auci)
            # test_auc

            cv_acc.append(best_acc)
            print('Best c:', best_c)
            print('Best g:', best_g)
            print('Best acc:', best_acc)
            print('test acc: ', test_acci)
            print('test auc', test_auci)
        print('Best cv acc all:', max(cv_acc))
        print('test_auc: ', test_auc)
        print('test_acc: ', test_acc)
        print('test_recall: ', test_recall)

        # ----- save ----------
        res[str(repeati)+'repeat_'+'cv_acc'] = cv_acc
        res[str(repeati)+'repeat_'+'test_acc'] = test_acc
        res[str(repeati)+'repeat_'+'test_precision'] = test_precision
        res[str(repeati)+'repeat_'+'test_f1'] = test_f1
        res[str(repeati)+'repeat_'+'test_auc'] = test_auc
        res[str(repeati)+'repeat_'+'test_recall'] = test_recall
        print(res)
    # res.to_csv('../fig2_6_result/pca_3geo_train_repeat_percent'+str(percenti)+'_top'+str(featnum)+'.csv')

