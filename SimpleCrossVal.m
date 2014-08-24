function [AccTest] = SimpleCrossVal(A,y_cs,method,nFoldToForget,OUTERkeys)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [AccTest] = SimpleCrossVal(A,y_cs,method,nFoldToForget,OUTERkeys)
%
% Performs a simple (not nested) cross-validation for graph-base 
% semi-supervised classification
%
% INPUT ARGUMENTS:
%  A:               nxn matrix, weighted undirected graph G containing n 
%                   nodes. represented by its symmetric adjacency matrix A.
%  y_cs:            nxm matrix, m binary indicator vectors y_c containing 
%                   as entries 1 for nodes belonging to the class whose 
%                   label index is c, and 0 otherwise.
%  method:          indicates which method will be used for the 
%                   classification
%  nFoldToForget:   indicates the labelling rate (the labelled nodes 
%                   represents (10-nFoldToForget)*10% of the data) and must 
%                   be an integer between 1 and 9 (included).
%  OUTERkeys:       controls which node is affected to which fold for the 
%                   cross-validation. is generate by the function
%                   "GenerateKeys".
%
% OUTPUT ARGUMENTS:
%  AccTest:         contains the f Accuracies where f is the number of
%                   folds.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nData,nClass] = size(y_cs);

nFold1 = 10; 
if nData < nFold1
    display('error')
end

% keeping a copy of y_cs and A
A_full = A; 
y_cs_full = y_cs; 

% to store the results 
AccTest = nan(1,nFold1);  

for f1 = 1:nFold1
    
    % re-initialisation of y_cs_full
    y_cs = y_cs_full; 
    
    % forget some labels
    Fs = [1:10 1:10];
    f1s = Fs(f1:f1+nFoldToForget-1);
    LabelToForget = false(1,nData);
    for i = f1s
        LabelToForget(OUTERkeys{i}) = true;
    end
    y_cs(LabelToForget,:) = 0;
    
    % call of the method
    [y_cs_new,SoS] = method(A,y_cs); 
    
    % keep only the to-estimate labels and prediction
    y_cs_new = y_cs_new(LabelToForget,:);
    y_cs_real = y_cs_full(LabelToForget,:);
    
    % test
    MatVerif = sum(y_cs_new == y_cs_real,2); 
    nCorrect = sum(MatVerif == nClass);
    
    AccTest(f1) = 100*nCorrect/sum(LabelToForget);
    
end

% AccTot = mean(AccTest);

end