function Surf =  main()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  Surf =  main()
% Starting function : obtains the accuracy for the labelling task (graph-
% based semi-supervised classification). the database can be added in db 
% and different methods can be added in MethodsList (as their parameters in 
% Params. nFoldToForget indicates the labelling rate (the labelled nodes 
% represents (10-nFoldToForget)*10% of the data) and must be an integer 
% between 1 and 9 (included). N allows to repeat the task N times with
% differents folds.
%
% OUTPUT ARGUMENTS:
%  Surf:    dxmxn matrix, big matrix containing the n accuracies for each
%           d databases and the m methods 
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
N = 20
nFold = 10; % never change this parameter !!!

for DB = [1:1]     
   % currently I am using only one dataset 
    db = {'mywisconsin_cocite'};          

    load(db{DB}) 
    
    A = Docr;
    y_cs = Classe2y_c(classeo);
    
    % Generate some folds which will remain the same
    [OUTERkeys,INNERkeys] = GenerateKeys(classeo,length(classeo),N,nFold);
    
    for nFoldToForget = 1:2:9 
        
        Surf = zeros(20,8,N);
        
        for n = 1:N
            
            display(n)
            
            for TODO =  [ 1:1 ] %   

                MethodsList = {...
                 
                    @BagOfP};           % Algo5    =>    8 

                Params = {{[10.^(-6:3)]}};                 	% Algo5

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                method = MethodsList{TODO}
                
                % run the cross-validation
                if numel(Params{TODO}) == 0
                    [AccTest] = SimpleCrossVal(Docr,y_cs,method,...
                        nFoldToForget,OUTERkeys(n,:)); 
                elseif numel(Params{TODO}) == 1
                    Param = cell2mat(Params{TODO});
                    [AccTest] = ...
                        DoubleCrossValA1Param(Docr,y_cs,Param,method,...
                        nFoldToForget,OUTERkeys,INNERkeys,n);
                else
                    display('Erreur : nParam doit �tre 0, 1')
                end

                AccTot = mean(AccTest);
                Surf(DB,TODO,n) = AccTot;
                
                % produce a report
                fid=fopen('res.txt','w');
                Display = strcat('Methode:',func2str(method),'\n');
                fprintf(fid,[Display]);

                Display = strcat('Database:',db{DB},'\n');
                fprintf(fid,[Display]);

                fprintf(fid, 'AccTest : %f\n', AccTot);
                fprintf(fid,'-------------\n');

            end
        end
      
    % save Surf for each database and each labelling rate    
    eval(['save RESULTS_DB' num2str(DB(1)) '_nFTF' num2str(nFoldToForget) '.mat Surf'])
    
    end
end

 fclose(fid);

end
