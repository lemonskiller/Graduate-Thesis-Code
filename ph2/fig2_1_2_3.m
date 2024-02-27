df=importdata('adni_p0.05_del_nagene.csv');

all_loc=df.data(2:end,1);
% diff_loc相邻位置之差
diff_loc=diff(all_loc);
mid_loc=median(diff_loc);
min_loc=min(diff_loc);
max_loc=max(diff_loc);
mean_loc=mean(diff_loc);

%% bar-diff loc分布 为美观，展示部分
figure
hist( diff_loc(diff_loc<5e5),50);
grid on

%% probe分布
figure
title('Cluster Analysis');
subplot(2,2,1)
%无分组
stem(all_loc(1:100),ones(size(all_loc(1:100))),'linewidth',1.5);
hold on %辅助
plot([all_loc(1)-100000,all_loc(105)],[0.2,0.2],'linewidth',2.5);

xlabel('(a) Original Distibution');
legend;
set(gca,'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

%过小分组
subplot(2,2,2)
stem(all_loc(1:6),ones(size(all_loc(1:6))),'linewidth',1.5);
hold on
stem(all_loc(7:10),ones(size(all_loc(7:10))),'linewidth',1.5);
hold on
stem(all_loc(11:21),ones(size(all_loc(11:21))),'linewidth',1.5);
hold on
stem(all_loc(22:31),ones(size(all_loc(22:31))),'linewidth',1.5);
hold on
stem(all_loc(32),ones(size(all_loc(32))),'linewidth',1.5);
hold on
stem(all_loc(33:40),ones(size(all_loc(33:40))),'linewidth',1.5);
hold on
stem(all_loc(41:50),ones(size(all_loc(41:50))),'linewidth',1.5);
hold on
stem(all_loc(51:69),ones(size(all_loc(51:69))),'linewidth',1.5);
hold on
stem(all_loc(70:88),ones(size(all_loc(70:88))),'linewidth',1.5);
hold on
stem(all_loc(89:98),ones(size(all_loc(89:98))),'linewidth',1.5);
hold on
stem(all_loc(99:100),ones(size(all_loc(99:100))),'linewidth',1.5);
hold on %辅助
plot([all_loc(1)-100000,all_loc(105)],[0.2,0.2],'linewidth',2.5);

xlabel('(b) Too Small Cluster Limit');
legend;
set(gca,'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

%理想分组
subplot(2,2,3)
stem(all_loc(1:6),ones(size(all_loc(1:6))),'linewidth',1.5);
hold on
stem(all_loc(7:31),ones(size(all_loc(7:31))),'linewidth',1.5);
hold on
stem(all_loc(32),ones(size(all_loc(32))),'linewidth',1.5);
hold on
stem(all_loc(33:50),ones(size(all_loc(33:50))),'linewidth',1.5);
hold on
stem(all_loc(51:69),ones(size(all_loc(51:69))),'linewidth',1.5);
hold on
stem(all_loc(70:98),ones(size(all_loc(70:98))),'linewidth',1.5);
hold on
stem(all_loc(99:100),ones(size(all_loc(99:100))),'linewidth',1.5);
hold on %辅助
plot([all_loc(1)-100000,all_loc(105)],[0.2,0.2],'linewidth',2.5);
% plot([all_loc(1)-100000,all_loc(105)],[0,0],'linewidth',2);
xlabel('(c) Ideal Cluster Limit');
legend;
set(gca,'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

%过大分组
subplot(2,2,4)
stem(all_loc(1:6),ones(size(all_loc(1:6))),'linewidth',1.5);
hold on
stem(all_loc(7:31),ones(size(all_loc(7:31))),'linewidth',1.5);
hold on
stem(all_loc(32:100),ones(size(all_loc(32:100))),'linewidth',1.5);
hold on %辅助
plot([all_loc(1)-100000,all_loc(105)],[0.2,0.2],'linewidth',2.5);

xlabel('(d) Oversized Cluster Limit');
legend;
set(gca,'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');


%% dbscan
% 范围肘点
figure
minpts=3;

plot(sort(diff_loc));
title('k-distance graph')
xlabel('Points sorted with 50th nearest distances')
ylabel('50th nearest distances')
grid

