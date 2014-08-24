function [OUTERkeys,INNERkeys] = GenerateKeys(classeo,nNodes,N,nFold)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [OUTERkeys,INNERkeys] = GenerateKeys(classeo,nNodes,N,nFold)
% Generates keys for a nested cross-validation for graph-base 
% semi-supervised classification. These keys are used to assign the nodes
% to different folds. The resulting folds will be stratified and as
% balanced as possible.
%
% INPUT ARGUMENTS:
%  classeo          nxm matrix, m binary indicator vectors y_c containing 
%                   as entries 1 for nodes belonging to the class whose 
%                   label index is c, and 0 otherwise.
%  nNodes:          the number of node of the graph. you can just put
%                   length(classeo)
%  N:        	    the number of different partition into fold to produce
%  nFold:           the number of fold. please put 10 here.
%
% OUTPUT ARGUMENTS:
%  OUTERkeys:       controls which node is affected to which fold for the 
%                   outer cross-validation. is generate by the function
%                   "GenerateKeys".
%  INNERkeys:       controls which node is affected to which fold for the 
%                   inner cross-validation. is generate by the function
%                   "GenerateKeys".
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OUTERkeys = cell(N,nFold);

for n=1:N
    
    % 1) randomise 
    
    key = randperm(nNodes);
    classe = y_c2Classe(classeo(key,:));
    
    % 2) sort
    
    [classeSorted,index] = sort(classe);
    key = key(index);
    
    % 3) add a destination cell
    
    tab = tabulate(classe);
    nInClass = tab(:,2);
    
    dest = [];
    for c = 1:length(nInClass)
        destc = zeros(1,nInClass(c));
        for i = 0:nInClass(c)-1
            destc(1,i+1) = mod(i,nFold)+1;
        end
        dest = [dest destc];
    end
    
    % 4) create the cells
    
    for fold = 1:nFold
        OUTERkeys{n,fold} = key(dest==fold);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now dividing again...

INNERkeys = cell(N,nFold,nFold);

for n=1:N
    
    for foldprim = 1:nFold
        
        % 1) randomise 
        
        key = OUTERkeys{n,foldprim};

        keyperm = randperm(length(key));
        
        key = key(keyperm);
        classe = classeo(key); 

        % 2) sort

        [classeSorted,index] = sort(classe);
        key = key(index);

        % 3) add a destination cell

        tab = tabulate(classe);
        nInClass = tab(:,2);

        dest = [];
        for c = 1:length(nInClass)
            destc = zeros(1,nInClass(c));
            for i = 0:nInClass(c)-1
                destc(1,i+1) = mod(i,nFold)+1;
            end
            dest = [dest destc];
        end
       
        % 4) create the cells

        for fold = 1:nFold
            INNERkeys{n,foldprim,fold} = key(dest==fold);
        end
    end
end

end

