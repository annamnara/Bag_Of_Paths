function [y_c] = Classe2y_c(Classe)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [y_c] = Classe2y_c(Classe)
% transforms a nx1 class membership vector (numbered 1 to c) into a nxc 
% matrix with each column containing a membership boolean to the c-th 
% class.
%
% INPUT ARGUMENTS:
%  Classe:  nx1 matrix, class membership vector of each n nodes (the n-th 
%           entry corresponds to the class number of the n-th node)
%
% OUTPUT ARGUMENTS:
%  y_cs:    nxc matrix, c binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[nlC,ncC] = size(Classe);
nClasse = max(Classe);
tab = tabulate(Classe);

y_c = zeros(nlC,nClasse);

for i = tab(:,1)'
    y_c(:,i) = Classe==i;
end

end

