# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


from sklearn.linear_model import LassoCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import matplotlib

# import sys
# libsvm_path='/home/data2/lemon/sgld_cmp/libsvm-3.24/python'
# sys.path.append(libsvm_path)


from svmutil import svm_train,svm_predict
from sklearn.model_selection import train_test_split

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
df=pd.read_csv(r"adni_p0.05_del_nagene.csv")
feat_sample=df.iloc[1:]
del feat_sample['IlmnID']
del feat_sample['Coordinate_36']
label=df.iloc[0,2:]
label=label.values
sample_feat=feat_sample.T.values
#----- 2 class ----
# label01=label[label != 2]
# sample_feat01= sample_feat[label != 2, :]

# train_sample_feat, train_label = sample_feat01, label01

#----- 3 class ----
train_sample_feat, train_label = sample_feat, label

#------training------
# 璁粌闆嗕氦鍙夐獙璇侊紝娴嬭瘯闆嗘祴璇?
percent=[1]#,0.5, 0.7, 0.9]
total_repeat = 10
for percenti in percent:
    res = pd.DataFrame()
    for repeati in range(1,1+total_repeat):

        X=train_sample_feat
        Y=train_label

        X[X < 0] = 0
        X[X > 1] = 1

        # ---- rf 特征筛选 ----

        from sklearn.ensemble import RandomForestRegressor

        # 调用LassoCV函数，并进行交叉验证，默认cv=3
        print('--- building rf  model ---')

        model = RandomForestRegressor(n_estimators=10)
        model.fit(X, Y)
        # 查看特征筛选情况

        # ----- libsvm分类 ------
        '''
        使用前百分比的策略进一步筛选的特征，采用libsvm训练
        '''
        desc_idx = np.argsort(-abs(np.array(model.feature_importances_)))

# --------- cv & test ----------
        cv_acc=[]
        for i in range(1,1+featnum):
            print('--- computing feature ',str(i),'---')
            select_x = X[:,desc_idx[:i]]
            best_c,best_g,best_acc=gridsearch_select(Y, select_x, fold)
            best_svmopt='-c '+str(best_c)+' -g '+str(best_g)+' -v '+str(fold)
            cv_acci = svm_train(Y,select_x,best_svmopt)
            cv_acc.append(cv_acci)

        # ----- save ----------
        allloc=df['Coordinate_36'][1:]
        desc_loc=allloc[1+desc_idx[:featnum]]

        res[str(repeati) + 'repeat_' + 'cv_acc'] = cv_acc
        res[str(repeati) + 'repeat_' + 'Coordinate_36'] = desc_loc.values
        res[str(repeati) + 'repeat_' + 'idx_start1'] = 1 + desc_idx[:featnum]
        print(res)

    # res.to_csv('../fig2_4_result/rf_adni3_repeat'+str(repeati)+'_percent'+str(percenti)+'_top'+str(featnum)+'10.csv')

