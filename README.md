# feature selection based on DNA methylation data
There are four paragraphs in my thesis, and the followings folders are codes corresponding to each paragraph. 
- **matlab_addpath:** my regression model for feature selection(without using any machine-learning package)
- **ph2:** feature selection process and result through my regression model
- **ph3:** feature selection process and result through traditional machine-learning methods
- **ph4:** feature selection process and result through deep-learning methods most of which are based on GCN 


```
.
├── README.md
├── add_loc_demo.ipynb
├── libsvm-3.24.rar
├── matlab_addpath
│   ├── all_accessment.m
│   ├── feature_selection.m
│   ├── gridsearchrbf.m
│   ├── mydbscan.m
│   ├── o2svm.m
│   ├── readkeys.m
│   ├── svm-predict.exe
│   ├── svm-train.exe
│   ├── svmscale.m
│   └── validation_function.m
├── mklaren.rar
├── ph2
│   ├── adni_p0.05_del_nagene.csv
│   ├── fig2_10_code
│   │   ├── fig2_10_L1_L21_RosmapGEO.m
│   │   ├── fig2_10_L1_L21_dst_RosmapGEO_linux.m
│   │   ├── fig2_10_L1_dst_RosmapGEO.m
│   │   ├── fig2_10_lasso_3geo_train_rosmap_test_linux.py
│   │   ├── searchCg
│   │   ├── searchCglog.txt
│   │   ├── svm-predict.exe
│   │   ├── svm-train.exe
│   │   ├── test_set
│   │   └── train_set
│   ├── fig2_10_result
│   │   ├── lasso_3geo_train_repeat_percent1_top30.csv
│   │   ├── rosmap_L1_L211.txt
│   │   ├── rosmap_L1_L2110.txt
│   │   ├── rosmap_L1_L212.txt
│   │   ├── rosmap_L1_L213.txt
│   │   ├── rosmap_L1_L214.txt
│   │   ├── rosmap_L1_L215.txt
│   │   ├── rosmap_L1_L216.txt
│   │   ├── rosmap_L1_L217.txt
│   │   ├── rosmap_L1_L218.txt
│   │   ├── rosmap_L1_L219.txt
│   │   ├── rosmap_L1_L21_dst.txt
│   │   ├── rosmap_L1_L21_dst10.txt
│   │   ├── rosmap_L1_L21_dst2.txt
│   │   ├── rosmap_L1_L21_dst3.txt
│   │   ├── rosmap_L1_L21_dst4.txt
│   │   ├── rosmap_L1_L21_dst5.txt
│   │   ├── rosmap_L1_L21_dst6.txt
│   │   ├── rosmap_L1_L21_dst7.txt
│   │   ├── rosmap_L1_L21_dst8.txt
│   │   ├── rosmap_L1_L21_dst9.txt
│   │   ├── rosmap_L1_dst.txt
│   │   ├── rosmap_L1_dst10.txt
│   │   ├── rosmap_L1_dst2.txt
│   │   ├── rosmap_L1_dst3.txt
│   │   ├── rosmap_L1_dst4.txt
│   │   ├── rosmap_L1_dst5.txt
│   │   ├── rosmap_L1_dst6.txt
│   │   ├── rosmap_L1_dst7.txt
│   │   ├── rosmap_L1_dst8.txt
│   │   └── rosmap_L1_dst9.txt
│   ├── fig2_11.svg
│   ├── fig2_11_12.m
│   ├── fig2_12.svg
│   ├── fig2_1_2_3.m
│   ├── fig2_3_code
│   │   ├── fig2_3_lasso_adni3.py
│   │   ├── fig2_3_net_adni3.py
│   │   ├── fig2_3_pca_adni3_linux.py
│   │   ├── fig2_3_rf_adni3_linux.py
│   │   ├── fig2_3_ridge_adni3.py
│   │   ├── fig2_3_sgld.m
│   │   ├── searchCg
│   │   ├── searchCglog.txt
│   │   ├── svm-predict.exe
│   │   ├── svm-train.exe
│   │   ├── test_set
│   │   └── train_set
│   ├── fig2_3_result
│   │   ├── ADNI_delnagene_L1_L21_dst---1.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---10.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---2.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---3.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---4.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---5.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---6.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---7.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---8.txt
│   │   ├── ADNI_delnagene_L1_L21_dst---9.txt
│   │   ├── ADNI_delnagene_L1_L21_dst3.txt
│   │   ├── lasso_adni3_repeat_percent1_lasso_top30.csv
│   │   ├── net_adni3_repeat_percent1_top30.csv
│   │   ├── pca_adni3_repeat_percent1_top30.csv
│   │   ├── rf_adni3_repeat1_percent1_top30_n10.csv
│   │   └── ridge_adni3_repeat_percent1_top30.csv
│   ├── fig2_4_5.m
│   ├── fig2_5_code
│   │   ├── fig2_5_L1_L21_dst_RosmapGEO.m
│   │   ├── fig2_5_lasso_3geo_train_rosmap_test_linux.py
│   │   ├── fig2_5_net_3geo_train_rosmap_test_linux.py
│   │   ├── fig2_5_pca_3geo_train_rosmap_test_linux.py
│   │   ├── fig2_5_rf_3geo_train_rosmap_test_fix.py
│   │   ├── fig2_5_ridge_3geo_train_rosmap_test_linux.py
│   │   ├── searchCg
│   │   ├── searchCglog.txt
│   │   ├── svm-predict.exe
│   │   ├── svm-train.exe
│   │   ├── test_set
│   │   └── train_set
│   ├── fig2_5_result
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst10.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst2.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst3.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst4.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst5.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst6.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst7.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst8.txt
│   │   ├── 12_1_0.4_0.2rosmap_L1_L21_dst9.txt
│   │   ├── lasso_3geo_train_repeat_percent1_top30.csv
│   │   ├── net_3geo_train_repeat_percent1_top30.csv
│   │   ├── pca_3geo_train_repeat_percent1_top30.csv
│   │   ├── rf_3geo_train_repeat_percent1_top30_n5.csv
│   │   └── ridge_3geo_train_repeat_percent1_top30.csv
│   ├── fig2_6_7.m
│   ├── fig2_7.svg
│   ├── fig2_8.m
│   ├── fig2_8.svg
│   ├── fig2_8_code
│   │   ├── fig2_8_L1_L21_ADNI3_linux.m
│   │   ├── fig2_8_L1_dst_ADNI3_linux.m
│   │   ├── fig2_8_l1_adni3.py
│   │   ├── fig2_8_sgld.m
│   │   ├── searchCg
│   │   ├── searchCglog.txt
│   │   ├── svm-predict.exe
│   │   ├── svm-train.exe
│   │   ├── test_set
│   │   └── train_set
│   ├── fig2_8_result
│   │   ├── 3322adni_L1_L21.txt
│   │   ├── 3322adni_L1_L2110.txt
│   │   ├── 3322adni_L1_L212.txt
│   │   ├── 3322adni_L1_L213.txt
│   │   ├── 3322adni_L1_L214.txt
│   │   ├── 3322adni_L1_L215.txt
│   │   ├── 3322adni_L1_L216.txt
│   │   ├── 3322adni_L1_L217.txt
│   │   ├── 3322adni_L1_L218.txt
│   │   ├── 3322adni_L1_L219.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_1.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_10.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_2.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_3.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_4.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_5.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_6.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_7.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_8.txt
│   │   ├── cmp_ADNI_delnagene_L1_dst_9.txt
│   │   ├── lasso_adni3_repeat_percent1_lasso_top30.csv
│   │   ├── 同fig2_3_ADNI_delnagene_L1_L21_dst3.txt
│   │   ├── 同fig2_3_ADNI_delnagene_L1_L21_dst_2.txt
│   │   ├── 同fig2_3_ADNI_delnagene_L1_L21_dst_3.txt
│   │   └── 同fig2_3_ADNI_delnagene_L1_L21_dst_4.txt
│   └── plt_acc.xlsx
├── ph2MATLAB部分重复10次.xlsx
├── ph3
│   ├── TADPOLE_D1_D2_del18_meth_1397.csv
│   ├── TADPOLE_D1_D2_del18_meth_1397_mci2.csv
│   ├── concat_class2
│   │   ├── adaboost_concat2mci_class2.py
│   │   ├── gbdt_concat2mci_class2.py
│   │   ├── knn_concat2mci_class2.py
│   │   ├── lr_concat2mci_class2.py
│   │   ├── rf_concat2mci_class2.py
│   │   ├── ridge_concat2mci_class2.py
│   │   ├── svm_concat2mci_class2.py
│   │   └── xgb_concat2mci_class2.py
│   ├── concat_class3
│   │   ├── adaboost_concat_class3.py
│   │   ├── gbdt_concat_class3.py
│   │   ├── knn_concat_class3.py
│   │   ├── lr_concat_class3.py
│   │   ├── rf_concat_class3.py
│   │   ├── ridge_concat_class3.py
│   │   ├── svm_concat_class3.py
│   │   └── xgb_concat_class3.py
│   ├── mk_class2.py
│   ├── mk_class3.py
│   ├── pca_class2
│   │   ├── adaboost_pca2mci_class2.py
│   │   ├── gbdt_pca2mci_class2.py
│   │   ├── knn_pca2mci_class2.py
│   │   ├── lr_pca2mci_class2.py
│   │   ├── rf_pca2mci_class2.py
│   │   ├── ridge_pca2mci_class2.py
│   │   ├── svm_pca2mci_class2.py
│   │   └── xgb_pca2mci_class2.py
│   └── pca_class3
│       ├── adaboost_pca_class3.py
│       ├── gbdt_pca_class3.py
│       ├── knn_pca_class3.py
│       ├── lr_pca_class3.py
│       ├── rf_pca_class3.py
│       ├── ridge_pca_class3.py
│       ├── svm_pca_class3.py
│       └── xgb_pca_class3.py
├── ph4
│   ├── AD-GCN-my-5fold-2classification-class2-demo.py
│   ├── AD-GCN-my-5fold-2classification-class2.py
│   ├── AD-GCN-my-5fold-3classification-class3-demo.py
│   ├── AD-GCN-my-5fold-3classification-class3.py
│   ├── Logger.py
│   ├── TADPOLE_D1_D2_del18_meth_1397.csv
│   ├── TADPOLE_D1_D2_del18_meth_1397_mci2.csv
│   ├── class2-c-gcn-clinic+sim.py
│   ├── class2-c-gcn-sim.py
│   ├── class2-ca-gcn-clinic+sim.py
│   ├── class2-gcn-sim.py
│   ├── class3-c-gcn-clinic+sim.py
│   ├── class3-c-gcn-sim.py
│   ├── class3-ca-gcn-clinic+sim.py
│   ├── class3-gcn-sim.py
│   ├── gcn
│   │   ├── __init__.py
│   │   ├── data
│   │   │   ├── ind.citeseer.allx
│   │   │   ├── ind.citeseer.ally
│   │   │   ├── ind.citeseer.graph
│   │   │   ├── ind.citeseer.test.index
│   │   │   ├── ind.citeseer.tx
│   │   │   ├── ind.citeseer.ty
│   │   │   ├── ind.citeseer.x
│   │   │   ├── ind.citeseer.y
│   │   │   ├── ind.cora.allx
│   │   │   ├── ind.cora.ally
│   │   │   ├── ind.cora.graph
│   │   │   ├── ind.cora.test.index
│   │   │   ├── ind.cora.tx
│   │   │   ├── ind.cora.ty
│   │   │   ├── ind.cora.x
│   │   │   ├── ind.cora.y
│   │   │   ├── ind.pubmed.allx
│   │   │   ├── ind.pubmed.ally
│   │   │   ├── ind.pubmed.graph
│   │   │   ├── ind.pubmed.test.index
│   │   │   ├── ind.pubmed.tx
│   │   │   ├── ind.pubmed.ty
│   │   │   ├── ind.pubmed.x
│   │   │   └── ind.pubmed.y
│   │   ├── inits.py
│   │   ├── layers.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── log
│   ├── my_result
│   │   ├── bl_with_meth.csv
│   │   ├── ca-gcn_with_meth.csv
│   │   ├── cagcn-clinic-sim_with_meth.csv
│   │   ├── cgcn-sim_with_meth.csv
│   │   ├── class2-cagcn-clinic-sim_with_meth.csv
│   │   ├── class2-cgcn-clinic-sim_with_meth.csv
│   │   ├── class2-cgcn-sim_with_meth.csv
│   │   ├── class2-gcn-sim_with_meth.csv
│   │   ├── class3-cagcn-clinic-sim_with_meth.csv
│   │   ├── class3-cgcn-clinic-sim_with_meth.csv
│   │   ├── class3-cgcn-sim_with_meth.csv
│   │   ├── class3_gcn_sim_with_meth.csv
│   │   └── gcn_sim_with_meth.csv
│   ├── predata
│   │   ├── TADPOLE_D1_D2_del18_meth_1397.csv
│   │   ├── TADPOLE_D1_D2_del18_meth_1397_mci2.csv
│   │   └── adni_adj_eye_mci2.py
│   ├── train_GCN.py
│   └── 程序复现说明+结果.docx
└── 运行方法+重复+结果.docx
```

# cite
- **tools:**
  - libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/
  - mklare: https://github.com/mstrazar/mklaren

- **databases:**
  - ADNI: https://adni.loni.usc.edu/
  - TCGA: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
  - GEO: https://www.ncbi.nlm.nih.gov/gds
