function o2svm(features,labels,filename)
% o2svm(features,labels,filename)
%
% O2SVM write change orginal feature table into file as SVM format 
% [lable index1:feature1 index2:feature2, ...]
%
% Input:
% features: original N*M feature table, N and M refer to numbers of samples and feaures respectively
% labels:   N*1 vector
% filename: user specified file name
%
% Copyright (c) 2012-03-29, Hua Tan
% skip the missing values revised by HT on 10/08/2014

[nr,nc]=size(features);
fid=fopen(filename,'w');
for i=1:nr
    fprintf(fid,'%d ',labels(i));
    for j=1:nc
        if isnan(features(i,j))
            continue;       % skip the missing values
        end
        fprintf(fid,'%d:%f ',j,features(i,j));
    end
    fprintf(fid,'\r\n');% ‘\r’回到行首，‘\n’到下一行
end
fclose(fid);

