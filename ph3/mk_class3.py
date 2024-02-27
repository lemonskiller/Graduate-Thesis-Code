import matplotlib.pyplot as plt
import seaborn as sns
import random

from scipy import interp
from itertools import cycle
import pandas as pd
import numpy as np
from sklearn import datasets
import pandas as pd

import math
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import metrics#, cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
# import mat4py
import numpy as np
from sklearn.svm import SVC

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from mklaren.mkl.alignf import Alignf
from mklaren.kernel.kernel import linear_kernel, poly_kernel, rbf_kernel,sigmoid_kernel
from mklaren.kernel.kinterface import Kinterface
from sklearn.model_selection import GridSearchCV

algo='mk_svm'
total_repeat=1
fs=18
# df = pd.read_csv(r'TADPOLE_D1_D2_del18_meth_1397.csv')
df = pd.read_csv(r'C:\Users\lemon\Documents\my_final\tadpole_challenge\TADPOLE_D1_D2_del18_meth_1397.csv')
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
with_without=['with','without']
# with_without=['with']

for with_withouti in with_without:
    print('-'*6)
    print(with_withouti)
    if with_withouti=='with':
        feat_ind = list(basic_ind)+list(grade_ind)+list(meth_ind)+list(mri_ind)
    else:
        feat_ind = list(basic_ind)+list(grade_ind)+list(mri_ind)
    sample_feat = df.iloc[:,feat_ind].values
    label = df.iloc[:,label_ind].values

    # 划分 val, test, trian: 随机等比例采样得 valid，test；bootstrap 采样得训练集
    
    label0_ind = np.where(label==0)
    label1_ind = np.where(label==1)
    label2_ind = np.where(label==2)

    test_per_class=50
    val_per_class=30

    random.seed(2020)
    random.shuffle(label0_ind)
    random.shuffle(label1_ind)
    random.shuffle(label2_ind)

    test_ind0=label0_ind[0][:test_per_class]
    test_ind1=label1_ind[0][:test_per_class]
    test_ind2=label2_ind[0][:test_per_class]
    val_ind0=label0_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind1=label1_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind2=label2_ind[0][test_per_class:test_per_class+val_per_class]
    remain_train_ind0=label0_ind[0][test_per_class+val_per_class:]
    remain_train_ind1=label1_ind[0][test_per_class+val_per_class:]
    remain_train_ind2=label2_ind[0][test_per_class+val_per_class:]

    # test val index
    test_ind=list(test_ind0)+list(test_ind1)+list(test_ind2)
    val_ind=list(val_ind0)+list(val_ind1)+list(val_ind2)
    # test
    test_sample_feat = sample_feat[test_ind,:]
    test_label = label[test_ind]
    # val
    val_sample_feat = sample_feat[val_ind,:]
    val_label = label[val_ind]
    # remain train
    all_train_sample_feat={}
    all_train_label={}
    all_model={}
    all_val_acc=[]
    train_per_class=min([len(remain_train_ind0),len(remain_train_ind1),len(remain_train_ind2)])

    for i in range(total_repeat):
        random.seed(i)
        random.shuffle(remain_train_ind0)
        random.shuffle(remain_train_ind1)
        random.shuffle(remain_train_ind2)

        train_ind0=remain_train_ind0[:train_per_class]
        train_ind1=remain_train_ind1[:train_per_class]
        train_ind2=remain_train_ind2[:train_per_class]
        train_ind=list(train_ind0)+list(train_ind1)+list(train_ind2)
        # random select traini
        train_sample_feati=sample_feat[train_ind,:]
        train_labeli=label[train_ind]

        all_train_sample_feat['fold_'+str(i)]=train_sample_feati
        all_train_label['fold_'+str(i)]=train_labeli


        K_exp  = Kinterface(data=train_sample_feati, kernel=rbf_kernel,kernel_args={"sigma": 2}) # RBF kernel 
        K_poly = Kinterface(data=train_sample_feati, kernel=poly_kernel)      # polynomial kernel with degree=3
        K_lin  = Kinterface(data=train_sample_feati, kernel=linear_kernel)                          # linear kernel
        K_sig = Kinterface(data=train_sample_feati, kernel=sigmoid_kernel)   

        model = Alignf(typ="linear")
        model.fit([K_exp, K_lin, K_poly, K_sig], np.array(train_labeli))

        mu = model.mu

        if with_withouti=='with':
            print('with meth kernel!')
            # exp lin poly
            combined_kernel1 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[1] * K_lin(x[:,-33:-15], y[:,-33:-15]) + mu[2] * K_poly(x[:,-15:], y[:,-15:])
            # exp lin sig
            combined_kernel2 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[1] * K_lin(x[:,-33:-15], y[:,-33:-15]) + mu[3] * K_sig(x[:,-15:], y[:,-15:])
            # exp sig lin
            combined_kernel3 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[3] * K_sig(x[:,-33:-15], y[:,-33:-15]) + mu[1] * K_lin(x[:,-15:], y[:,-15:])

            # exp poly sig 
            combined_kernel4 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[2] * K_poly(x[:,-33:-15], y[:,-33:-15]) + mu[3] * K_sig(x[:,-15:], y[:,-15:])

            parameters={'kernel':[combined_kernel1,combined_kernel2,combined_kernel3,combined_kernel4]}
        else:
            print('without meth kernel!')
            # exp lin 
            combined_kernel1 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[1] * K_lin(x[:,-33:-15], y[:,-33:-15])
            # exp  sig
            combined_kernel2 = lambda x, y: \
                mu[0] * K_exp(x[:,:-33], y[:,:-33]) + mu[3] * K_sig(x[:,-33:-15], y[:,-33:-15])

            parameters={'kernel':[combined_kernel1,combined_kernel2]}

        model = GridSearchCV(SVC(probability=True),parameters,cv=5,scoring='accuracy')

        model.fit(train_sample_feati, train_labeli)
        all_model['fold_'+str(i)]=model
        # save model
        val_pre = model.predict(val_sample_feat).round()
        # val_pro = model.predict_proba(val_sample_feat)
        val_acci = accuracy_score(val_label,val_pre)
        all_val_acc.append(val_acci)

    # select model
    max_ind = all_val_acc.index(max(all_val_acc))
    best_model=all_model['fold_'+str(max_ind)]
    print('max val acc: ',all_val_acc[max_ind])
    print('-'*6)

    # test
    test_pre = best_model.predict(test_sample_feat).round()
    # auc
    labels=np.unique(label)
    test_auc = roc_auc_score(test_label, label_binarize(test_pre, classes=labels),average='macro',multi_class = 'ovr')
    # 获取分类报告
    cr = classification_report(test_label, test_pre, output_dict=True)
    cr['auc']=test_auc
    # pd.DataFrame(cr).to_csv('./kernel/'+algo+'_'+with_withouti+'_meth.csv')


    ############### 混淆矩阵 #####################
    sns.set(font='Times New Roman',font_scale=1,style="ticks")  # ticks white

    fig = plt.figure()
    ax = fig.add_subplot(111)

    conf_mat = confusion_matrix(test_label,test_pre)

    if with_withouti=='with':
        cmapi='summer'
    else:
        cmapi='Blues'

    h = sns.heatmap(conf_mat,annot=True,xticklabels=['NC','MCI','AD'],yticklabels=['NC','MCI','AD'], fmt='g',cmap=cmapi,annot_kws={'size':26,'weight':'bold','family':'Times New Roman'},linewidths=1,cbar=True,square=True) #画热力图

    # sns.heatmap(conf_mat/sum(sum(conf_mat)),annot=True,xticklabels=['NC','MCI','AD'],yticklabels=['NC','MCI','AD'],cmap='Blues',annot_kws={'size':20,'weight':'bold','family':'Times New Roman'},linewidths=0,cbar=False,square=True) #画热力图

    ax.set_title('Confusion Matrix',fontsize=fs+2,weight='bold',family='Times New Roman') #标题
    ax.set_xlabel('Predict Label',fontsize=fs,weight='bold',family='Times New Roman') #x轴
    ax.set_ylabel('True Label',fontsize=fs,weight='bold',family='Times New Roman') #y轴

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs,weight='bold',family='Times New Roman')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs,weight='bold',family='Times New Roman', rotation=0)

    ax.grid(True, which='minor', linestyle='-',color='b')
    plt.show()

    # fig.savefig('./kernel/'+algo+'_heatmap_'+with_withouti+'_meth.svg', dpi=600, bbox_inches='tight')


    ################ ROC ################
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2

    y_score=model.predict_proba(test_sample_feat)
    # Binarize the output
    y_test = label_binarize(test_label, classes=[0, 1, 2])
    n_classes = y_test.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    ############ Plot all ROC curves ###############
    # plt.figure()
    
    fig = plt.figure() 
    
    ax = fig.add_subplot(111)
    

    if with_withouti=='with':
        plt.style.use('seaborn-ticks')
    else:
        plt.style.use('classic')
    sns.set(font='Times New Roman', style="ticks",font_scale=1) 

    # multi class
    lstyles = cycle(['-', '--', '-'])
    class_names = ['NC','MCI','AD']
    for i,lstyle in zip(range(n_classes),lstyles):
        plt.plot(fpr[i], tpr[i], lw=lw,linestyle=lstyle,
                label='ROC Curve of {0} (area = {1:0.2f})'
                ''.format(class_names[i], roc_auc[i]))

    # macro micro
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-Average ROC Curve (Area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            linestyle='-',marker='^', linewidth=2.5)

    plt.plot(fpr["macro"], tpr["macro"],
            label='Macro-Average ROC Curve (Area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            linestyle='-',marker='s', linewidth=2.5)



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fs,weight='bold',family='Times New Roman')
    plt.ylabel('True Positive Rate',fontsize=fs,weight='bold',family='Times New Roman')
    plt.title('Multi-Class ROC Curve',fontsize=fs+3,weight='bold',family='Times New Roman')
    plt.legend(fontsize=16,loc='best')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs,weight='bold',family='Times New Roman')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs,weight='bold',family='Times New Roman', rotation=0)

    plt.show()
    # fig.savefig('./kernel/'+algo+'_roc_'+with_withouti+'_meth.svg', dpi=600, bbox_inches='tight')