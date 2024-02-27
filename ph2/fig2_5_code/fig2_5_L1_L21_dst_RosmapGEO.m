% ROSMAP 测试集 独立验证
% GEO 训练集 交叉验证
% 超参数——网格搜索L1、L21、distance
% 聚类参数——坐标值的中位数附近
% 特征数——固定为30
clear;%清除工作区
close all;%关闭当前所有窗口
%% 导入数据
disease_name = 'ROSMAP_GEO';
% rawdata = importdata('D:/work/lasso-yang+zhao/data wash/R-select/R_PRAD_TCGA_n502c499_GEO_n75c75.csv');
rawdata = importdata('ROSMAP_497_GEO_488_del_nagene.csv');

outname = '../fig2_5_result/12_1_0.4_0.2rosmap_L1_L21_dst10.txt';
test_num = 497;

% 原始数据
feature_name = rawdata.textdata(3:end,1)'; %特征名字/甲基化探针名字
sample = rawdata.data(2:end,2:end)'; %样本数据,样本*特征
label = rawdata.data(1,2:end)'; %NC MCI AD
probe_coordinate = rawdata.data(2:end,1); %探针的坐标位置

%%
% 测试集
test_label = label(1:test_num);
test_sample = sample(1:test_num,:);
% 训练集 
train_label = label(test_num+1:end);
train_sample = sample(test_num+1:end,:);
%% 网格搜参
% 超参数
eps = median(diff(probe_coordinate)); %dbscan聚类参数
pick_num = 30; %特征数
step = 0.2; %搜参步长
train_acc = []; %交叉验证准确率
test_acc = []; %独立验证准确率
select_feature_index_used = []; %存储规则，每一列对应每一组参数
parameter_memory = []; %存储参数值，每一列对应每一组参数
for dst=0.2 %step:step:0.2
    for L1=1
        for L21=0.4
            parameter_memory = [parameter_memory,[L1;L21;dst]];
            %% 特征选择
            [weight,select_feature,~] = feature_selection(train_sample,train_label,feature_name,probe_coordinate,L21,L1,dst,eps);
            [~,~,select_feature_index] = intersect(select_feature,feature_name,'stable'); %根据特征名字求出特征对应的索引
            new_select_feature_index = select_feature_index(1:pick_num); %确定特征数
            select_feature_index_used = [select_feature_index_used,new_select_feature_index];
            %% 交叉验证——对训练集的样本数据进行特征缩放处理
            scale_method = 1; %svmscale输入要求:（1）x:样本数*特征数的矩阵（2）method：1缩放到[-1,1];0:缩放到[0,1]
            [scale_train_x,~] = svmscale(train_sample(:,new_select_feature_index),scale_method); 
            %% 交叉验证——用初选特征通过网格搜索确定SVM参数
            o2svm(scale_train_x,train_label,'searchCg'); %转换成工具包支持的文件格式
            [bestC,bestg,CVaccu,~] = gridsearchrbf('searchCg',[],[],[],[],[]);
            train_acc = [train_acc; CVaccu]; %最佳参数情况下的五折交叉验证准确率
            svmopt = ['-t 2 -g ',num2str(bestg),' -c ',num2str(bestC)]; %最优化参数
            %% 训练模型
            o2svm(scale_train_x,train_label,'train_set') %转换成工具包支持的文件格式
            [train_y,train_x] = libsvmread('train_set');
            model = svmtrain(train_y,train_x,svmopt);
            %% 独立验证——对测试集的样本数据进行特征缩放处理
            scale_method = 1; %svmscale输入要求:（1）x:样本数*特征数的矩阵（2）method：1缩放到[-1,1];0:缩放到[0,1]
            [scale_test_x,~] = svmscale(test_sample(:,new_select_feature_index),scale_method);
            %% 独立验证——测试集准确率
            o2svm(scale_test_x,test_label,'test_set') % 转换成工具包支持的文件格式
            [test_y,test_x] = libsvmread('test_set');
            [predict_label,accuracy,~] = svmpredict(test_y,test_x,model);%求测试集准确率
            test_acc = [test_acc; accuracy(1)];
        end
    end
end
%% 最大交叉验证准确率，此时的特征数
[~, p] = max(train_acc); %p为网格搜参下，最大交叉验证准确率对应的循环次数/索引
select_parameter = parameter_memory(:,p);
select_feature_index = select_feature_index_used(:,p);
%% 导出最优参数和对应的探针索引（探针按坐标大小升序排列，再求所选探针的索引）
[~,coordinate_sort_idx] = sort(probe_coordinate);
[~,~,top_30_probe_index] = intersect(feature_name(select_feature_index),feature_name(coordinate_sort_idx),'stable');
fid = fopen(outname, 'a+');
fprintf(fid, '%s\t%s\t%s\t%s\n', ('L1'),('L21'),('dst'),('top 30 probe index'));
fprintf(fid, '%f\t%f\t%f\t', (select_parameter(1)),(select_parameter(2)),(select_parameter(3)));
for i=1:pick_num-1
    fprintf(fid, '%d\t', (top_30_probe_index(i)));
end
fprintf(fid, '%d\n', (top_30_probe_index(pick_num)));
fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ('feature num'),('train acc'),('test acc'),...
                         ('f1'),('auc'),('acc'),('recall'),('precision'),('FPR'),('TNR'),('TPR')); %导出评价指标
fclose(fid);
%% 最优参数下，特征数1-30
train_acc_30 = zeros(pick_num,1);
test_acc_30 = zeros(pick_num,1);
for k=1:pick_num
    new_select_feature_index = select_feature_index(1:k); %确定特征数
    %% 交叉验证——对训练集的样本数据进行特征缩放处理
    scale_method = 1; %svmscale输入要求:（1）x:样本数*特征数的矩阵（2）method：1缩放到[-1,1];0:缩放到[0,1]
    [scale_train_x,~] = svmscale(train_sample(:,new_select_feature_index),scale_method); 
    %% 交叉验证——用初选特征通过网格搜索确定SVM参数
    o2svm(scale_train_x,train_label,'searchCg'); %转换成工具包支持的文件格式
    [bestC,bestg,CVaccu,~] = gridsearchrbf('searchCg',[],[],[],[],[]);
    train_acc_30(k,1) = CVaccu;
    svmopt = ['-t 2 -g ',num2str(bestg),' -c ',num2str(bestC)]; %最优化参数
    %% 训练模型
    o2svm(scale_train_x,train_label,'train_set') %转换成工具包支持的文件格式
    [train_y,train_x] = libsvmread('train_set');
    model = svmtrain(train_y,train_x,svmopt);
    %% 独立验证——对测试集的样本数据进行特征缩放处理
    scale_method = 1; %svmscale输入要求:（1）x:样本数*特征数的矩阵（2）method：1缩放到[-1,1];0:缩放到[0,1]
    [scale_test_x,~] = svmscale(test_sample(:,new_select_feature_index),scale_method);
    %% 独立验证——测试集准确率
    o2svm(scale_test_x,test_label,'test_set') % 转换成工具包支持的文件格式
    [test_y,test_x] = libsvmread('test_set');
    [predict_label,accuracy,~] = svmpredict(test_y,test_x,model);%求测试集准确率
    test_acc_30(k,1) = accuracy(1);
    %% 其他统计学指标
    [f1,auc,acc,recall,precision,FPR,TNR,TPR] = all_accessment(predict_label,test_y);
    %% 导出数据
    fid = fopen(outname, 'a+');
%     fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ('feature num'),('train acc'),('test acc'),...
%                          ('f1'),('auc'),('acc'),('recall'),('precision'),('FPR'),('TNR'),('TPR')); %导出评价指标
    fprintf(fid, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', (k),(CVaccu),(accuracy(1)),...
                         (f1),(auc),(acc),(recall),(precision),(FPR),(TNR),(TPR)); %导出评价指标
    fclose(fid);
end
%% 导出最优参数下，1-30中的最佳准确率
[max_train_acc,feature_num] = max(train_acc_30);
max_test_acc = test_acc_30(feature_num,1); %不是最大的测试集准确率，是交叉验证准确率最大时，对应的测试集准确率
fid = fopen(outname, 'a+');
fprintf(fid, '%d\t%f\t%f\t%s\n', (feature_num),(max_train_acc),(max_test_acc),('max_train_acc'));
fclose(fid);