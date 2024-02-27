import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import random


from sklearn.decomposition import PCA
fs=18
algo='gbdt'
total_repeat=5
df = pd.read_csv('../TADPOLE_D1_D2_del18_meth_1397_mci2.csv')
# df = pd.read_csv(r'C:\Users\lemon\Documents\my_final\tadpole_challenge\TADPOLE_D1_D2_del18_meth_1397_mci.csv')
# df = pd.read_csv(r'C:\Users\lemon\Documents\my_final\pygcn-master\pygcn\TADPOLE_D1_D2_del18_meth_1397_mci2.csv')

# 性别
gender_ind=5
# 年龄
age_ind=4
# APOE4
apoe_ind=6
# label
label_ind=18
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
# with_without=['with']
with_without=['with','without']

for with_withouti in with_without:
    print('-'*6)
    print(with_withouti)
    label = df.iloc[:,label_ind].values

    # pca->10
    sample_feat_clinic = df.iloc[:,list(basic_ind)+list(grade_ind)].values
    sample_feat_mri = df.iloc[:,list(mri_ind)].values
    sample_feat_meth = df.iloc[:,list(meth_ind)].values

    pca = PCA(n_components=10)
    sample_feat_clinic_pca = pca.fit_transform(sample_feat_clinic)
    sample_feat_mri_pca = pca.fit_transform(sample_feat_mri)
    sample_feat_meth_pca = pca.fit_transform(sample_feat_meth)
    
    sample_feat_clinic_pca = MinMaxScaler().fit_transform(sample_feat_clinic_pca)
    sample_feat_mri_pca = MinMaxScaler().fit_transform(sample_feat_mri_pca)
    sample_feat_meth_pca = MinMaxScaler().fit_transform(sample_feat_meth_pca)

    sample_feat=np.add(sample_feat_clinic_pca,sample_feat_mri_pca)
    if with_withouti=='with':
        sample_feat=np.add(sample_feat,sample_feat_meth_pca)


    # 划分 val, test, trian: 随机等比例采样得 valid，test；bootstrap 采样得训练集
    random.seed(2020)
    label1_ind = np.where(label==1)
    label2_ind = np.where(label==2)

    test_per_class=30
    val_per_class=10
    test_ind1=label1_ind[0][:test_per_class]
    test_ind2=label2_ind[0][:test_per_class]

    val_ind1=label1_ind[0][test_per_class:test_per_class+val_per_class]
    val_ind2=label2_ind[0][test_per_class:test_per_class+val_per_class]

    remain_train_ind1=label1_ind[0][test_per_class+val_per_class:]
    remain_train_ind2=label2_ind[0][test_per_class+val_per_class:]

    # test val index
    test_ind=list(test_ind1)+list(test_ind2)
    val_ind=list(val_ind1)+list(val_ind2)
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
    all_scaler={}
    all_val_acc=[]
    train_per_class=min([len(remain_train_ind1),len(remain_train_ind2)])

    for i in range(total_repeat):
        random.seed(i)

        # 从中随机抽取train_per_class个
        train_ind1 = random.sample(list(remain_train_ind1), train_per_class)
        train_ind2 = random.sample(list(remain_train_ind2), train_per_class)
        train_ind=list(train_ind1)+list(train_ind2)
        # random select traini
        train_sample_feati=sample_feat[train_ind,:]
        train_labeli=label[train_ind]

        ########## min max norm #############
        scaler = MinMaxScaler().fit(train_sample_feati)
        train_sample_feati = scaler.transform(train_sample_feati)
        val_sample_feati = scaler.transform(val_sample_feat)

        all_train_sample_feat['fold_'+str(i)]=train_sample_feati
        all_train_label['fold_'+str(i)]=train_labeli

        if with_withouti=='without':
            model = GradientBoostingClassifier(n_estimators=300,random_state=2020) # without 12; with
        else:
            model = GradientBoostingClassifier(n_estimators=800,random_state=2020) # without 12; with
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
    print(pd.DataFrame(cr))
    # pd.DataFrame(cr).to_csv('./pca_add2mci/'+algo+'_'+with_withouti+'_meth.csv')

    ############### 混淆矩阵 #####################
    sns.set(font='Times New Roman',font_scale=1,style="ticks")  # ticks white

    fig = plt.figure()
    ax = fig.add_subplot(111)

    conf_mat = confusion_matrix(test_label,test_pre)

    if with_withouti=='with':
        cmapi='summer'
    else:
        cmapi='Blues'

    h = sns.heatmap(conf_mat,annot=True,xticklabels=['MCI-MCI','MCI-AD'],yticklabels=['MCI-MCI','MCI-AD'], fmt='g',cmap=cmapi,annot_kws={'size':26,'weight':'bold','family':'Times New Roman'},linewidths=1,cbar=True,square=True) #画热力图

    # sns.heatmap(conf_mat/sum(sum(conf_mat)),annot=True,xticklabels=['NC','MCI','AD'],yticklabels=['NC','MCI','AD'],cmap='Blues',annot_kws={'size':20,'weight':'bold','family':'Times New Roman'},linewidths=0,cbar=False,square=True) #画热力图

    ax.set_title('Confusion Matrix',fontsize=fs+2,weight='bold',family='Times New Roman') #标题
    ax.set_xlabel('Predict Label',fontsize=fs,weight='bold',family='Times New Roman') #x轴
    ax.set_ylabel('True Label',fontsize=fs,weight='bold',family='Times New Roman') #y轴

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs,weight='bold',family='Times New Roman')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs,weight='bold',family='Times New Roman', rotation=0)

    ax.grid(True, which='minor', linestyle='-',color='b')
    plt.show()

    # fig.savefig('./pca_add2mci/'+algo+'_heatmap_'+with_withouti+'_meth.svg', dpi=600, bbox_inches='tight')