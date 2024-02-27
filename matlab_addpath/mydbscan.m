function [IDX, isnoise]=mydbscan(X,epsilon,MinPts,max_gap)
    C=0;
    n=size(X,1);
    IDX=-ones(n,1);% noise:-1
    D=pdist2(X,X);
    visited=false(n,1);
    isnoise=false(n,1);
    for i=1:n
        if ~visited(i)
            visited(i)=true;
            
            Neighbors=RegionQuery(i);
            if numel(Neighbors)<MinPts
                % X(i,:) is NOISE
                isnoise(i)=true;
            else
                C=C+1;
                ExpandCluster(i,Neighbors,C,max_gap);
            end
            
        end
    end
    
    function ExpandCluster(i,Neighbors,C,max_gap)
%         if ~visited(i)
%             visited(i)=true;
%             IDX(i)=C;
%         end
        IDX(i)=C;
        k = 1;
        while true
            j = Neighbors(k);
            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j);
                if numel(Neighbors2)>=MinPts 
                    min_loci=min(X(find(IDX==C)));
                    max_loci=max(X(find(IDX==C)));
%                     for ii=1:length(Neighbors2)
                        if max([abs(min_loci-X(j)),abs(max_loci-X(j))])<=max_gap% my add &-
                            Neighbors=[Neighbors j];   %#ok
                        else %范围太大，要分裂
                            ExpandCluster(j,Neighbors,C+1,max_gap);
                        end
%                     end
                end
            
            end
            if IDX(j)==-1
                IDX(j)=C;
            end
            
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
    end
    
    function Neighbors=RegionQuery(i)
        Neighbors=find(D(i,:)<=epsilon);
    end
end