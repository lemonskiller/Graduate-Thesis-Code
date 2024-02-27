% ROSMAP ���Լ� ������֤
% GEO ѵ���� ������֤
% ������������������L1��L21��distance
% ���������������ֵ����λ������
% �����������̶�Ϊ30
clear;%���������
close all;%�رյ�ǰ���д���
%% ��������
disease_name = 'ROSMAP_GEO';
% rawdata = importdata('D:/work/lasso-yang+zhao/data wash/R-select/R_PRAD_TCGA_n502c499_GEO_n75c75.csv');
rawdata = importdata('ROSMAP_497_GEO_488_del_nagene.csv');

outname = '../fig2_5_result/12_1_0.4_0.2rosmap_L1_L21_dst10.txt';
test_num = 497;

% ԭʼ����
feature_name = rawdata.textdata(3:end,1)'; %��������/�׻���̽������
sample = rawdata.data(2:end,2:end)'; %��������,����*����
label = rawdata.data(1,2:end)'; %NC MCI AD
probe_coordinate = rawdata.data(2:end,1); %̽�������λ��

%%
% ���Լ�
test_label = label(1:test_num);
test_sample = sample(1:test_num,:);
% ѵ���� 
train_label = label(test_num+1:end);
train_sample = sample(test_num+1:end,:);
%% �����Ѳ�
% ������
eps = median(diff(probe_coordinate)); %dbscan�������
pick_num = 30; %������
step = 0.2; %�Ѳβ���
train_acc = []; %������֤׼ȷ��
test_acc = []; %������֤׼ȷ��
select_feature_index_used = []; %�洢����ÿһ�ж�Ӧÿһ�����
parameter_memory = []; %�洢����ֵ��ÿһ�ж�Ӧÿһ�����
for dst=0.2 %step:step:0.2
    for L1=1
        for L21=0.4
            parameter_memory = [parameter_memory,[L1;L21;dst]];
            %% ����ѡ��
            [weight,select_feature,~] = feature_selection(train_sample,train_label,feature_name,probe_coordinate,L21,L1,dst,eps);
            [~,~,select_feature_index] = intersect(select_feature,feature_name,'stable'); %���������������������Ӧ������
            new_select_feature_index = select_feature_index(1:pick_num); %ȷ��������
            select_feature_index_used = [select_feature_index_used,new_select_feature_index];
            %% ������֤������ѵ�������������ݽ����������Ŵ���
            scale_method = 1; %svmscale����Ҫ��:��1��x:������*�������ľ���2��method��1���ŵ�[-1,1];0:���ŵ�[0,1]
            [scale_train_x,~] = svmscale(train_sample(:,new_select_feature_index),scale_method); 
            %% ������֤�����ó�ѡ����ͨ����������ȷ��SVM����
            o2svm(scale_train_x,train_label,'searchCg'); %ת���ɹ��߰�֧�ֵ��ļ���ʽ
            [bestC,bestg,CVaccu,~] = gridsearchrbf('searchCg',[],[],[],[],[]);
            train_acc = [train_acc; CVaccu]; %��Ѳ�������µ����۽�����֤׼ȷ��
            svmopt = ['-t 2 -g ',num2str(bestg),' -c ',num2str(bestC)]; %���Ż�����
            %% ѵ��ģ��
            o2svm(scale_train_x,train_label,'train_set') %ת���ɹ��߰�֧�ֵ��ļ���ʽ
            [train_y,train_x] = libsvmread('train_set');
            model = svmtrain(train_y,train_x,svmopt);
            %% ������֤�����Բ��Լ����������ݽ����������Ŵ���
            scale_method = 1; %svmscale����Ҫ��:��1��x:������*�������ľ���2��method��1���ŵ�[-1,1];0:���ŵ�[0,1]
            [scale_test_x,~] = svmscale(test_sample(:,new_select_feature_index),scale_method);
            %% ������֤�������Լ�׼ȷ��
            o2svm(scale_test_x,test_label,'test_set') % ת���ɹ��߰�֧�ֵ��ļ���ʽ
            [test_y,test_x] = libsvmread('test_set');
            [predict_label,accuracy,~] = svmpredict(test_y,test_x,model);%����Լ�׼ȷ��
            test_acc = [test_acc; accuracy(1)];
        end
    end
end
%% ��󽻲���֤׼ȷ�ʣ���ʱ��������
[~, p] = max(train_acc); %pΪ�����Ѳ��£���󽻲���֤׼ȷ�ʶ�Ӧ��ѭ������/����
select_parameter = parameter_memory(:,p);
select_feature_index = select_feature_index_used(:,p);
%% �������Ų����Ͷ�Ӧ��̽��������̽�밴�����С�������У�������ѡ̽���������
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
                         ('f1'),('auc'),('acc'),('recall'),('precision'),('FPR'),('TNR'),('TPR')); %��������ָ��
fclose(fid);
%% ���Ų����£�������1-30
train_acc_30 = zeros(pick_num,1);
test_acc_30 = zeros(pick_num,1);
for k=1:pick_num
    new_select_feature_index = select_feature_index(1:k); %ȷ��������
    %% ������֤������ѵ�������������ݽ����������Ŵ���
    scale_method = 1; %svmscale����Ҫ��:��1��x:������*�������ľ���2��method��1���ŵ�[-1,1];0:���ŵ�[0,1]
    [scale_train_x,~] = svmscale(train_sample(:,new_select_feature_index),scale_method); 
    %% ������֤�����ó�ѡ����ͨ����������ȷ��SVM����
    o2svm(scale_train_x,train_label,'searchCg'); %ת���ɹ��߰�֧�ֵ��ļ���ʽ
    [bestC,bestg,CVaccu,~] = gridsearchrbf('searchCg',[],[],[],[],[]);
    train_acc_30(k,1) = CVaccu;
    svmopt = ['-t 2 -g ',num2str(bestg),' -c ',num2str(bestC)]; %���Ż�����
    %% ѵ��ģ��
    o2svm(scale_train_x,train_label,'train_set') %ת���ɹ��߰�֧�ֵ��ļ���ʽ
    [train_y,train_x] = libsvmread('train_set');
    model = svmtrain(train_y,train_x,svmopt);
    %% ������֤�����Բ��Լ����������ݽ����������Ŵ���
    scale_method = 1; %svmscale����Ҫ��:��1��x:������*�������ľ���2��method��1���ŵ�[-1,1];0:���ŵ�[0,1]
    [scale_test_x,~] = svmscale(test_sample(:,new_select_feature_index),scale_method);
    %% ������֤�������Լ�׼ȷ��
    o2svm(scale_test_x,test_label,'test_set') % ת���ɹ��߰�֧�ֵ��ļ���ʽ
    [test_y,test_x] = libsvmread('test_set');
    [predict_label,accuracy,~] = svmpredict(test_y,test_x,model);%����Լ�׼ȷ��
    test_acc_30(k,1) = accuracy(1);
    %% ����ͳ��ѧָ��
    [f1,auc,acc,recall,precision,FPR,TNR,TPR] = all_accessment(predict_label,test_y);
    %% ��������
    fid = fopen(outname, 'a+');
%     fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', ('feature num'),('train acc'),('test acc'),...
%                          ('f1'),('auc'),('acc'),('recall'),('precision'),('FPR'),('TNR'),('TPR')); %��������ָ��
    fprintf(fid, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', (k),(CVaccu),(accuracy(1)),...
                         (f1),(auc),(acc),(recall),(precision),(FPR),(TNR),(TPR)); %��������ָ��
    fclose(fid);
end
%% �������Ų����£�1-30�е����׼ȷ��
[max_train_acc,feature_num] = max(train_acc_30);
max_test_acc = test_acc_30(feature_num,1); %�������Ĳ��Լ�׼ȷ�ʣ��ǽ�����֤׼ȷ�����ʱ����Ӧ�Ĳ��Լ�׼ȷ��
fid = fopen(outname, 'a+');
fprintf(fid, '%d\t%f\t%f\t%s\n', (feature_num),(max_train_acc),(max_test_acc),('max_train_acc'));
fclose(fid);