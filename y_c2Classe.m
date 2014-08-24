function [Classe] = y_c2Classe(y_c)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [Classe] = Classe2y_c(y_c)
% transforms a nxc matrix with each column containing a membership boolean 
% to the c-th class into a nx1 class membership vector (numbered 1 to c).
%
% INPUT ARGUMENTS:
%  y_cs:    nxc matrix, c binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%
% OUTPUT ARGUMENTS:
%
%  Classe:  nx1 matrix, class membership vector of each n nodes (the n-th 
%           entry corresponds to the class number of the n-th node)
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[useless,nC] = size(y_c);
Classe = y_c*[1:nC]';

end

