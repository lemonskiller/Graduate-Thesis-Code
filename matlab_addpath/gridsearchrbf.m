function [bestC,bestg,CVaccu,results]=gridsearchrbf(datafile,log2c,log2g,v,svmopt,refine)
% 改编自↓
%[bestC,bestg,CVaccu,results]=gridsearchrbf(datafile,log2c,log2g,v,refine)
%
% implement the grid.py by matlab for RBF kernel:exp(-gamma*|u-v|^2)
% see also gridsearchpoly.m, gridsearchsigm.m
% 
% Input:
%   datafile---name of file (typically text file) storing the labels and features data (use
%              o2svm.m to create such kind of file)
%              each row = [lable index1:feature1 index2:feature2, ...]
%   log2c    ---set the range of c (default:[-1 15 2])
%              [begin,end,step] -- c_range = 2^{begin,...,begin+k*step,...,end} 
%   log2g    ---set the range of g (default:[3 -15 -2])
%              [begin,end,step] -- g_range = 2^{begin,...,begin+k*step,...,end} 
%   v        ---set the fold number v (default: 5-fold cross validation)
%   svmopt   ---other svm options for svm-train.exe in LIBSVM
%   refine   ---indicator of refining grid search (default: 0, will not do refining)
%
% Output:
%   bestC,bestg,CVaccu  --- searched optimal parameters with highest v-fold
%                           cross validation (CV) accuracy
%   results             --- all parameter combinations and corresponding CV accuracy
% 
% if begin==end in the log2c or log2g, will not grid with that particular parameter
%
% parameters correspond to those used in grid.py
% for options for svm-train.exe, see http://www.csie.ntu.edu.tw/~cjlin/libsvm/
%
% by Hua Tan 5/10/2014
% revised on 10/31/2014

tic
if ~exist('svm-train.exe','file')
    error('tool unavailable: download svm-train.exe from LIBSVM and put it in the current directory');
end

if nargin<2 || isempty(log2c)
    log2c=[-1 15 2];     %
end

if nargin<3 || isempty(log2g)
    log2g=[7 -15 -2];    %
end

if nargin<4 || isempty(v)
    v=5;
end

if nargin<6
    svmopt='';
end

if nargin<5 || isempty(refine)
    refine=0;   %do not do finer grid search
end

cbegin  =log2c(1); cend    =log2c(2); cstep   =log2c(3);
rangeC=2.^(cbegin:cstep:cend);

gbegin  =log2g(1); gend    =log2g(2); gstep   =log2g(3);
rangeg=2.^(gbegin:gstep:gend);

logfile=[datafile 'log.txt'];
Cgs=zeros(0);
accuracy=zeros(0);
counter=1;
svmopto=svmopt;
for ic=1:length(rangeC)
    for ig=1:length(rangeg)
        Cgst=[rangeC(ic) rangeg(ig)];
        Cgs(counter,:)=Cgst;
        svmopt=[svmopto ' -g ' num2str(rangeg(ig)) ' -c ' num2str(rangeC(ic)) ' -v ' num2str(v) ' ']; %default to rbf kernel, no need to set '-t 2'
        eval(['!svm-train ' svmopt datafile ' > ' logfile]);  %把字符串当作命令来执行
        logs=readkeys(logfile,char(10),3);
        accuracy(counter)=processlog(logs);
        disp(['[C g ACCU] = ' num2str([Cgst accuracy(counter)])]);
        counter=counter+1;
    end
end

if size(Cgs,1)~=length(accuracy)
    disp('Warning: something wrong..');
end

[maxaccu indmax]=max(accuracy);

bestC=Cgs(indmax,1);
bestg=Cgs(indmax,2);
CVaccu=maxaccu;
results=[Cgs accuracy'];
disp(['best [C g ACCU] = ' num2str([bestC bestg CVaccu])]);

if refine   %do refine
    rr=max(0.5,min(0.75,refine));   %between 0.5 and .75
    disp(['finer griddig: rr=' num2str(rr)]);
    bc=log2(bestC);bg=log2(bestg);
    nlog2c=[bc-rr bc+rr 0.25];  %new log2c
    nlog2g=[bg+rr bg-rr -0.25];
    [nbestC, nbestg, nCVaccu, nresults]=gridsearchrbf(datafile,nlog2c,nlog2g,v,svmopt);
    bestC=nbestC;bestg=nbestg;CVaccu=nCVaccu;
    results=[results;nresults];
end
toc
disp(datestr(now))

function accuracy=processlog(logfile)   %to use, just un-comment this function
accuracy=zeros(0);
counter=1;
for i=1:length(logfile)
    if strfind(logfile{i},'Cross Validation Accuracy')
        istart=strfind(logfile{i},'=');
        iend=strfind(logfile{i},'%');
        accuracy(counter,1)=str2double(logfile{i}(istart+2:iend-1));
        counter=counter+1;
    end
end