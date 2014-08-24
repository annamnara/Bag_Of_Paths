function [AccTest] = DoubleCrossValA1Param(A,y_cs,AValider,method,...
    nFoldToForget,OUTERkeys,INNERkeys,n)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [AccTest] = DoubleCrossValA1Param(A,y_cs,AValider,method,...
%    nFoldToForget,OUTERkeys,INNERkeys,n)
% Performs a nested cross-validation for graph-base semi-supervised 
% classification
%
% INPUT ARGUMENTS:
%  A:               nxn matrix, weighted undirected graph G containing n 
%                   nodes. represented by its symmetric adjacency matrix A.
%  y_cs:            nxm matrix, m binary indicator vectors y_c containing 
%                   as entries 1 for nodes belonging to the class whose 
%                   label index is c, and 0 otherwise.
%  AValider:        indicates all the value of the parameter to try for the
%                   inner cross-validation.
%  method:          indicates which method will be used for the 
%                   classification
%  nFoldToForget:   indicates the labelling rate (the labelled nodes 
%                   represents (10-nFoldToForget)*10% of the data) and must 
%                   be an integer between 1 and 9 (included).
%  OUTERkeys:       controls which node is affected to which fold for the 
%                   outer cross-validation. is generate by the function
%                   "GenerateKeys".
%  INNERkeys:       controls which node is affected to which fold for the 
%                   inner cross-validation. is generate by the function
%                   "GenerateKeys".
%  n:               indicates which key must be used from the two previous 
%                   input arguments. useful if the classifiaction task is
%                   repeted more than one times (and is just 1 otherwise)
%
% OUTPUT ARGUMENTS:
%  AccTest:         contains the f Accuracies where f is the number of
%                   folds.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nData,nClass] = size(y_cs); 
V = length(AValider); 

nFold1 = 10; 
if nData < nFold1
    display('error')
end

% keeping a copy of y_cs and A
A_full = A; 
y_cs_full = y_cs; 

% to store the results
AccVal = nan(nFold1,nFold1,V); 
AccMean = nan(nFold1,V); 
AccTest = nan(1,nFold1);  

for f1 = 1:nFold1
    
    Fs = [1:10 1:10];
    f1s = Fs(f1:f1+nFoldToForget-1);
    LabelToForgetOUTER = false(1,nData);
    for i = f1s
        LabelToForgetOUTER(OUTERkeys{i}) = true;
    end
    
    for v = 1:V 
        
        Param = AValider(v);
        
        for f2 = 1:nFold1
            
            % re-initialisation of y_cs_full
            y_cs = y_cs_full; 
            
            % forget which labels ?
            f2s = Fs(f2:f2+nFoldToForget-1);
            % LabelToForget = false(1,nData);
            LabelToForgetOUT = false(1,nData);
            LabelToForgetIN = false(1,nData);
            
            % the labels forgotten because of OUTER cross-val
            for i = f1s
                for j = 1:nFold1
                    LabelToForgetOUT(INNERkeys{n,i,j}) = true;
                end
            end
            
            % the labels forgotten because of INNER cross-val
            for j = 1:nFold1
                if sum(j ~= f1s)==numel(f1s)
                    LabelToForgetIN(INNERkeys{n,j,f2s(1)}) = true;
                end
            end
            
            LabelToForget = or(LabelToForgetOUT,LabelToForgetIN);
            
            % forget some labels
            y_cs(LabelToForget,:) = 0; 
            
            % call of the method
            [y_cs_new,SoS] = method(A,y_cs,Param); 
            
            % keep only the to-estimate labels and prediction
            y_cs_new = y_cs_new(LabelToForgetIN,:);
            y_cs_real = y_cs_full(LabelToForgetIN,:);
          
            % test
            MatVerif = sum(y_cs_new == y_cs_real,2); 
            nCorrect = sum(MatVerif == nClass); 
            AccVal(f1,f2,v) = 100*nCorrect/sum(LabelToForgetIN);
        end
        
        AccMean(f1,v) = AccVal(f1,:,v)*ones(nFold1,1)/nFold1;
          
    end 
    
    % maximize mean accuracy
    [AccMax,BestOne] = max(AccMean(f1,:));
    
    display(['gagnant de la validation : ' num2str(AValider(BestOne))])
    Winner = AValider(BestOne); 
    
    % re-initialisation of y_cs_full
    y_cs = y_cs_full; 
            
    % forget some labels
    y_cs(LabelToForgetOUTER,:) = 0;
    
    % call of the method
    [y_cs_new,SoS] = method(A,y_cs,Winner); 
    
    % keep only the to-estimate labels and prediction
    y_cs_new = y_cs_new(LabelToForgetOUTER,:); 
    y_cs_real = y_cs_full(LabelToForgetOUTER,:); 
    
    % test
    MatVerif = sum(y_cs_new == y_cs_real,2); 
    nCorrect = sum(MatVerif == nClass); 
    
    AccTest(f1) = 100*nCorrect/sum(LabelToForgetOUTER);
    
end

% AccTot = mean(AccTest);

end
