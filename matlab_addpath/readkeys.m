function keys=readkeys(filename,spliters,opt)
% keys=readkeys(filename,spliters,opt)
%
% to read keys from a file, keys are split by 'spliters'
% Input:
%   filename    ---name of file to read 
%   spliters    ---delimiter for data reading (default: spliters=[char(9) ' '])
%                   spliters=char(10)--read the file line by line            
%   opt         ---specify the data structure 
%                   1--cell array of cells (default)
%                   2--m*n cell matrix
%                   3--1*K cell vector
% by Hua Tan on 01/09/2013
% revised on 07/26/2013

fid=fopen(filename,'r');

if fid<0 
    error(['fail to open ' filename]);
end

if nargin<2 || isempty(spliters)
    spliters=[char(9) ' ']; 
end

if nargin<3 
    opt=1; 
end

if strcmpi(spliters,char(10))
    opt=3;
end

keys=cell(0);
counter=1;
while ~feof(fid)
    line=fgetl(fid);
    
    if isempty(line)   %skip empty lines
        continue;
    end

    switch opt
        case 1
            domain=getoken(line,spliters);
            keys{counter}=domain;   %cell array of cells
        case 2
            domain=getoken(line,spliters);
            keys(counter,:)=domain; %m*n cell matrix
        case 3
            keys{counter}=line;     %1*K cell vector
        otherwise
            error('input error: opt');
    end
    
    counter=counter+1;
    
    if ~mod(counter,10000) 
        disp(num2str(counter)); 
    end
end
fclose(fid);

function tokens=getoken(str,spliters)
tokens=cell(0);
counter=1;
[token remain]=strtok(str,spliters);
while ~isempty(token)
    tokens{counter}=token;
    counter=counter+1;
    [token remain]=strtok(remain,spliters);
end 
