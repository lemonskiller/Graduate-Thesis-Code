function [weights,features,funVal] = feature_selection(x,y,feature_name,feature_coordinate,lambda1,lambda2,lambda3,epsilon)
%% ���ݷ����Ԥ����
% y ������ǩ
% x ��������
% lambda1 ��lasso
% lambda2 lasso
% lambda3 �������
% epsilon dbscan�������
% weights ɸѡ����Ȩ������Ҫ������
% features ɸѡ��������
% featureIndex ɸѡ������������������x�е�����
% funVal Ŀ�꺯���ĵ���ֵ
%% ��̽�밴��������
[~,M] = size(x);%�����������
[sort_feature_coordinate,sort_label] = sort(feature_coordinate);%��̽������갴�������У�sort_labelΪ���ص�����
X = x(:,sort_label);%���ŵ���������
sort_feature_name = feature_name(:,sort_label);%���ŵİ�֢������
%% ��̽����������ܶȾ���
% group_index = dbscan(sort_feature_coordinate,epsilon,3);%�ܶȾ���
group_index = mydbscan(sort_feature_coordinate,epsilon,3,2*epsilon);%�ܶȾ���
% group_index = average_group(sort_feature_coordinate,floor(M/15));%ƽ�����飬������֤��lasso��ϡ����lasso��Ӱ��
group_num = max(group_index);%��������ֵ��Ϊ���������
p_num = zeros(group_num,1);%����ÿ���̽��������p1,...,p_m
group_feature_index = cell(group_num,1);%����ÿ��̽�������
j = 1;%�����ж�ĳ��̽�����ڵڼ���
feature_index = [];%�������гɴص�̽�������
feature_index_j = [];%ѭ������ʱ����ÿ��̽�������
for i=1:M
    %group_index=-1��Ϊ������
    if(group_index(i,1)==j)
        p_num(j) = p_num(j)+1;
        feature_index_j = [feature_index_j;i];
        feature_index = [feature_index;i];
    end
    if i<M
        if(group_index(i+1,1)==j+1)
            group_feature_index{j} = feature_index_j;
            feature_index_j = [];
            j = j+1;
        end
    end
end
group_feature_index{j} = feature_index_j;%�������һ���̽������
% P = sum(p_num);%�ɴص�̽���������
% sort_feature_name = sort_feature_name(:,feature_index);%ɸѡ���ɴ�̽��֮���̽������
%% �������� d
d = zeros(M,M);%�������
% coordinate = sort_feature_coordinate(feature_index,:);%ɸѡ���ɴ�̽��֮���̽������
coordinate = sort_feature_coordinate;
sigma = std(coordinate);%���׼��
for i=1:M
    for j=1:M
        tmp = coordinate(i,1)-coordinate(j,1);
        d(i,j) = exp(-tmp*tmp/(2*sigma*sigma));
    end
end
%% �������ݷ���
x = cell(group_num,1);%��̽���������������ݵķ��飬ÿ������������У�ΪN��̽�������У��ֱ�Ϊp1,...,p_m
for i=1:group_num
    x{i} = X(:,group_feature_index{i});
end
%% Ȩ�ط���
w = cell(group_num,1);
for i=1:group_num
    w{i} = ones(p_num(i),1)/1000;
end
%% objective function
R = y;%����ѭ�����ʼ�в���
for i=1:group_num
    R = R-x{i}*w{i};
end 
%�����½��㷨
beta0 = ones(M,1)/1000;%������Ȩ��д��һ��������
beta0(group_index==-1) = 0; 
J = norm(R,2)*norm(R,2)/2+lambda2*sum(abs(beta0));%��ʼĿ�꺯��
for i=1:group_num
    J = J + lambda1 *norm(w{i},2);
end
for i=1:M
    for j=1:M
        J = J+lambda3*d(i,j)*(beta0(i,1)-beta0(j,1))*(beta0(i,1)-beta0(j,1))/4;
    end
end%��ʼĿ�꺯��
delta = J;%�����ж�����
funVal = [];%�洢������Ŀ�꺯����ֵ
beta = zeros(M,1);%������Ȩ��д��һ��������
iteration = 0;
while iteration<20
    beta_lastIteration = beta;
    for k=1:group_num
        %������Ȩ��д��һ��������
        for i=1:group_num
            beta(group_feature_index{i})=w{i};
        end%����betaֵ
        beta(group_index==-1) = 0; 
    
        R_k = R+x{k}*w{k};%����R_k
        S3 = zeros(p_num(k),1);%��w{k}=0�����������һ��������йصĲ���
        for i=1:p_num(k)
            t = group_feature_index{k}(i,1);%��t��ʹ�æ�t=w{k}(i,1),i��S3(i,1)��i
            tmp_matrix = d(:,t).*beta;
            S3(i,1) = -sum(tmp_matrix)+sum(tmp_matrix(group_feature_index{k}(1):group_feature_index{k}(p_num(k)),1));
        end
        
        condition1 = zeros(p_num(k),1);%�ж�����Ȩ���Ƿ�Ϊ0��һ������
        for i=1:p_num(k)
            Ctmp = x{k}'*R_k-lambda3*S3;
            if(Ctmp(i,1)<-lambda2)
                condition1(i,1) = Ctmp(i,1)+lambda2;
            else
                if(Ctmp(i,1)>lambda2)
                    condition1(i,1) = Ctmp(i,1)-lambda2;
                else
                    condition1(i,1) = 0;
                end
            end
        end
        if(norm(condition1,2)<=lambda1)%�ж�����Ȩ�Ƿ�Ϊ0
            w{k}=zeros(p_num(k,1),1);%����Ȩ��Ϊ0
        else
            for j=1:p_num(k)
                R_kj = R_k;%����/����R_kj
                for i=1:p_num(k)
                    if(i~=j)
                        R_kj = R_kj-x{k}(:,i).*w{k}(i,1);
                    end
                end
                
                t = group_feature_index{k}(j,1);%��t��ʹ�æ�t=w{k}(j,1)
                dtmp = sum(d(:,t).*beta)-d(t,t)*beta(t,1);%��w{k}(j,1)=0����������е�һ��������йصĲ���
                if(abs(x{k}(:,j)'*R_kj+lambda3*dtmp)<=lambda2)
                    w{k}(j,1)=0;%��k���j��Ȩ��Ϊ0
                else
                    w2 = x{k}(:,j)'*x{k}(:,j)+lambda1/norm(w{k},2)+lambda3*(sum(d(:,t))-d(t,t));%��w{k}(j,1)�ķ�ĸ
                    if(x{k}(:,j)'*R_kj+lambda3*dtmp>lambda2)
                        w1 = x{k}(:,j)'*R_kj+lambda3*dtmp-lambda2;
                        w{k}(j,1) = w1/w2;
                    end
                    if(x{k}(:,j)'*R_kj+lambda3*dtmp<-lambda2)
                        w1 = x{k}(:,j)'*R_kj+lambda3*dtmp+lambda2;
                        w{k}(j,1) = w1/w2;
                    end
                end
                R = R_kj - x{k}(:,j).*w{k}(j,1);%����R
                R_k = R+x{k}*w{k};%����R_k
                beta(t,1) = w{k}(j,1);%����Ȩ��������
            end
        end
    end
    %���µ�������
    iteration = iteration+1;
    disp(['======��������',num2str(iteration),'======']);
    %����Ŀ�꺯��ֵ
    Jtmp = J;%��ʱ�洢���ε���֮ǰ��Ŀ�꺯��ֵ
    J = norm(R,2)*norm(R,2)/2+lambda2*sum(abs(beta));%���±��ε������Ŀ�꺯��ֵ
    for i=1:group_num
        J = J + lambda1*norm(w{i},2);
    end
    for i=1:M
        for j=1:M
            J = J+lambda3*d(i,j)*(beta(i,1)-beta(j,1))*(beta(i,1)-beta(j,1))/4;
        end
    end
    delta = abs(J-Jtmp);%����delta
%     delta = round(delta*10000)/10000;
    funVal = [funVal;J];%�洢���ε������Ŀ�꺯��ֵ
     %�ж��Ƿ���ȨֵΪNaN
    if sum(isnan(beta))
        beta= beta_lastIteration;
        break
    end 
end
%% ���
flag = beta ~= 0;
weight_num = sum(flag(:));%���в�Ϊ0��Ȩ�صĸ���������ѡ���Ȩ�صĸ���
select_weight = zeros(weight_num,1);
select_feature = cell(weight_num,1);
j = 1;
for i=1:M
    if(flag(i,1))
        select_weight(j,1) = beta(i,1);
        select_feature{j} = sort_feature_name{i};
        j = j+1;
    end
end
[~,sort_index2] = sort(abs(select_weight),'descend');
weights = select_weight(sort_index2,:);
features = select_feature(sort_index2,:);
