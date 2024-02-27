function [smat scoef]=svmscale(mat,method,coef)
% [smat scoef]=svmscale(mat,method,coef)
%
% linear scale features (on columns) by the formula: smat=(smat-A)/B
% Input:
%   mat         ---M*N matrix, M=number of samples,N=number of features
%   method      ---designate the scaling range
%                   if method=1 or [], scale mat to [-1 1] with B=range(mat)/2, A=max(mat)-B
%                   elseif method=0, scale mat to [0 1] with A=min(mat),B=range(mat)=max(mat)-min(mat)
%                   otherwise, A=mean(mat), B=std(mat)
%
%   coef(opt)   ---2*N matrix, N=number of features
%                   if coef is designated (nargin>2), A and B can be extracted from coef
%
% from now on, normusr.m, normusr2.m,normusrtest.m will not be useful anymore
% by Hua Tan on 10/07/2014


if nargin<2 || isempty(method)
    method=1;
end

smat=mat;
scoef=[];

if nargin>2
   if size(coef,2)~=size(mat,2)
            disp('length of range should be equal to column number of mat');
            return;
   else
       A=coef(1,:);
       B=coef(2,:);
   end  
else
    if method==1      %default:linear scale to [-1 1]
        B=range(mat)/2;
        A=max(mat)-B;
    elseif method==0  %linear scale to [0 1]
        A=min(mat);
        B=range(mat);
    else              %nonlinear scale to ~N(0,1),not usually used, because this scaling will probably change the data structure
        A=mean(mat);
        B=std(mat);
    end
end
% A(~B)=0;B(~B)=1;  %keep the columns with zero range unchanged [i.e., let A=0, and B=1 at columns where B==0, then (x-A)/B==(x-0)/1==x], by HT on 10/09/2014
% looks like we do not need the above operation,actually, if B==0, that is,
% instances take the same value (or just miss the value) at that feature,
% then this feature is uninformative for the classification,and the following formula (x-A)/B will
% produce a NaN at that feature, fine!
scoef=[A;B];
A=repmat(A,size(mat,1),1);
B=repmat(B,size(mat,1),1);
smat=(mat-A)./B;



