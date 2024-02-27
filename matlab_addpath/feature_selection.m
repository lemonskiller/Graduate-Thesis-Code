function [weights,features,funVal] = feature_selection(x,y,feature_name,feature_coordinate,lambda1,lambda2,lambda3,epsilon)
%% 数据分组等预处理
% y 样本标签
% x 样本数据
% lambda1 组lasso
% lambda2 lasso
% lambda3 距离参数
% epsilon dbscan聚类参数
% weights 筛选出的权按照重要性排序
% features 筛选出的特征
% featureIndex 筛选出的特征在样本数据x中的索引
% funVal 目标函数的迭代值
%% 对探针按距离排序
[~,M] = size(x);%最初的特征数
[sort_feature_coordinate,sort_label] = sort(feature_coordinate);%将探针的坐标按升序排列，sort_label为返回的索引
X = x(:,sort_label);%重排的样本数据
sort_feature_name = feature_name(:,sort_label);%重排的癌症特征名
%% 对探针坐标进行密度聚类
% group_index = dbscan(sort_feature_coordinate,epsilon,3);%密度聚类
group_index = mydbscan(sort_feature_coordinate,epsilon,3,2*epsilon);%密度聚类
% group_index = average_group(sort_feature_coordinate,floor(M/15));%平均分组，用于验证组lasso和稀疏组lasso的影响
group_num = max(group_index);%结果的最大值即为分组的数量
p_num = zeros(group_num,1);%储存每组的探针数量，p1,...,p_m
group_feature_index = cell(group_num,1);%储存每组探针的索引
j = 1;%用于判断某个探针属于第几组
feature_index = [];%储存所有成簇的探针的索引
feature_index_j = [];%循环中暂时储存每组探针的索引
for i=1:M
    %group_index=-1即为噪声点
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
group_feature_index{j} = feature_index_j;%保存最后一组的探针索引
% P = sum(p_num);%成簇的探针的总数量
% sort_feature_name = sort_feature_name(:,feature_index);%筛选出成簇探针之后的探针名字
%% 求距离矩阵 d
d = zeros(M,M);%距离矩阵
% coordinate = sort_feature_coordinate(feature_index,:);%筛选出成簇探针之后的探针坐标
coordinate = sort_feature_coordinate;
sigma = std(coordinate);%求标准差
for i=1:M
    for j=1:M
        tmp = coordinate(i,1)-coordinate(j,1);
        d(i,j) = exp(-tmp*tmp/(2*sigma*sigma));
    end
end
%% 样本数据分组
x = cell(group_num,1);%按探针分组进行样本数据的分组，每组的样本数（行）为N，探针数（列）分别为p1,...,p_m
for i=1:group_num
    x{i} = X(:,group_feature_index{i});
end
%% 权重分组
w = cell(group_num,1);
for i=1:group_num
    w{i} = ones(p_num(i),1)/1000;
end
%% objective function
R = y;%用于循环求初始残差项
for i=1:group_num
    R = R-x{i}*w{i};
end 
%坐标下降算法
beta0 = ones(M,1)/1000;%将所有权重写成一个列向量
beta0(group_index==-1) = 0; 
J = norm(R,2)*norm(R,2)/2+lambda2*sum(abs(beta0));%初始目标函数
for i=1:group_num
    J = J + lambda1 *norm(w{i},2);
end
for i=1:M
    for j=1:M
        J = J+lambda3*d(i,j)*(beta0(i,1)-beta0(j,1))*(beta0(i,1)-beta0(j,1))/4;
    end
end%初始目标函数
delta = J;%用于判断收敛
funVal = [];%存储迭代中目标函数的值
beta = zeros(M,1);%将所有权重写成一个列向量
iteration = 0;
while iteration<20
    beta_lastIteration = beta;
    for k=1:group_num
        %将所有权重写成一个列向量
        for i=1:group_num
            beta(group_feature_index{i})=w{i};
        end%更新beta值
        beta(group_index==-1) = 0; 
    
        R_k = R+x{k}*w{k};%更新R_k
        S3 = zeros(p_num(k),1);%求w{k}=0的条件语句中一个与距离有关的参数
        for i=1:p_num(k)
            t = group_feature_index{k}(i,1);%求t，使得βt=w{k}(i,1),i即S3(i,1)的i
            tmp_matrix = d(:,t).*beta;
            S3(i,1) = -sum(tmp_matrix)+sum(tmp_matrix(group_feature_index{k}(1):group_feature_index{k}(p_num(k)),1));
        end
        
        condition1 = zeros(p_num(k),1);%判断整组权重是否为0的一个向量
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
        if(norm(condition1,2)<=lambda1)%判断整组权是否为0
            w{k}=zeros(p_num(k,1),1);%整组权重为0
        else
            for j=1:p_num(k)
                R_kj = R_k;%计算/更新R_kj
                for i=1:p_num(k)
                    if(i~=j)
                        R_kj = R_kj-x{k}(:,i).*w{k}(i,1);
                    end
                end
                
                t = group_feature_index{k}(j,1);%求t，使得βt=w{k}(j,1)
                dtmp = sum(d(:,t).*beta)-d(t,t)*beta(t,1);%求w{k}(j,1)=0的条件语句中的一个与距离有关的参数
                if(abs(x{k}(:,j)'*R_kj+lambda3*dtmp)<=lambda2)
                    w{k}(j,1)=0;%第k组第j个权重为0
                else
                    w2 = x{k}(:,j)'*x{k}(:,j)+lambda1/norm(w{k},2)+lambda3*(sum(d(:,t))-d(t,t));%求w{k}(j,1)的分母
                    if(x{k}(:,j)'*R_kj+lambda3*dtmp>lambda2)
                        w1 = x{k}(:,j)'*R_kj+lambda3*dtmp-lambda2;
                        w{k}(j,1) = w1/w2;
                    end
                    if(x{k}(:,j)'*R_kj+lambda3*dtmp<-lambda2)
                        w1 = x{k}(:,j)'*R_kj+lambda3*dtmp+lambda2;
                        w{k}(j,1) = w1/w2;
                    end
                end
                R = R_kj - x{k}(:,j).*w{k}(j,1);%更新R
                R_k = R+x{k}*w{k};%更新R_k
                beta(t,1) = w{k}(j,1);%更新权重向量β
            end
        end
    end
    %更新迭代次数
    iteration = iteration+1;
    disp(['======迭代次数',num2str(iteration),'======']);
    %更新目标函数值
    Jtmp = J;%暂时存储本次迭代之前的目标函数值
    J = norm(R,2)*norm(R,2)/2+lambda2*sum(abs(beta));%更新本次迭代后的目标函数值
    for i=1:group_num
        J = J + lambda1*norm(w{i},2);
    end
    for i=1:M
        for j=1:M
            J = J+lambda3*d(i,j)*(beta(i,1)-beta(j,1))*(beta(i,1)-beta(j,1))/4;
        end
    end
    delta = abs(J-Jtmp);%更新delta
%     delta = round(delta*10000)/10000;
    funVal = [funVal;J];%存储本次迭代后的目标函数值
     %判断是否有权值为NaN
    if sum(isnan(beta))
        beta= beta_lastIteration;
        break
    end 
end
%% 输出
flag = beta ~= 0;
weight_num = sum(flag(:));%所有不为0的权重的个数，即被选择的权重的个数
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
