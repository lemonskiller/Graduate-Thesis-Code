function [f1 auc acc recall precision FPR TNR TPR]=all_accessment(pre_label,true_label)
pos=find(pre_label==1);
neg=find(pre_label==0);
auc=0;
if length(pos)~=0 & length(neg)~=0 
    for i=1:length(pos)
        for j=1:length(neg)
            if pre_label(pos(i))>pre_label(neg(j))
                auc=auc+1;
            end
            if pre_label(pos(i))==pre_label(neg(j))
                auc=auc+0.5;
            end
        end
    end
    auc=auc/(length(pos)*length(neg));
end


t_in_prel_id=find(pre_label==1);
t_in_l_id=find(true_label==1);
[tp_id,]=intersect(t_in_prel_id,t_in_l_id);
tp=length(tp_id);
fn=length(t_in_l_id)-tp;
fp=length(t_in_prel_id)-tp;
tn=sum(pre_label~=1)-fn;%
TNR=tn/(fp+tn);
FPR=fp/(fp+tn);
TPR=tp/(tp+fn);
acc=(tp+tn)/(tp+fp+tn+fn);
if tp==0
    precision=0;
    recall=0;
else
    precision=tp/(tp+fp);
    recall=tp/(tp+fn);
end
f1=2*precision*recall/(precision+recall);



