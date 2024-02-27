import numpy as np
import pandas as pd


from sklearn.linear_model import LassoCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import matplotlib

import sys
libsvm_path='/home/lemon/sgld/libsvm-3.24/python'
sys.path.append(libsvm_path)

from svmutil import svm_train,svm_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
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
#----- 3 class ----
train_sample_feat, train_label = sample_feat, label

#------training------
# 璁粌闆嗕氦鍙夐獙璇侊紝娴嬭瘯闆嗘祴璇?
percent=[1]#,0.5, 0.7, 0.9]
total_repeat = 1
for percenti in percent:
    res = pd.DataFrame()
    for repeati in range(1,1+total_repeat):

        X=train_sample_feat
        Y=train_label

        X[X < 0] = 0
        X[X > 1] = 1

        # ------ select features by pca ----------------
        from sklearn.decomposition import PCA

        # --------- cv & test ----------
        cv_acc = []

        for i in range(1, 1 + featnum):
            print('--- making feature ', str(i), '---')
            pca = PCA(n_components=i)
            select_x = pca.fit_transform(X)
            best_c, best_g, best_acc = gridsearch_select(Y, select_x, fold)
            cv_acc.append(best_acc)
            print('Best acc:', best_acc)

        # ----- save ----------
        res[str(repeati) + 'repeat_' + 'cv_acc'] = cv_acc
        print(res)

    # res.to_csv('../fig2_4_result/pca_adni3_repeat_percent'+str(percenti)+'_top'+str(featnum)+'.csv')

