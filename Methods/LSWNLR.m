function [U,SoS] = LSWNLR(A,y_cs,lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [U,SoS]=LSWNLR(A,y_cs,lambda))
% Algorithm 1* A simple regularization framework for labeling the nodes of 
% a weighted undirected graph G. Normalized Laplacian is used. 
%
% INPUT ARGUMENTS:
%  A:       nxn matrix, weighted undirected graph G containing n nodes.
%           represented by its symmetric adjacency matrix A.
%  y_cs:    nxm matrix, m binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%  lambda:  the (scalar) value of the regularization parameter, lambda 
%           must be > 0.
%
% OUTPUT ARGUMENTS:
%  U:       nxm matrix, membership matrix containing the membership of 
%           each node i to class k, u_ik.
%  SoS:     nxm matrix, indicate the sum of similarities of each node i
%           to class k, sos_ik.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if lambda <= 0
    display('error : lambda must be > 0')
end

[n,m] = size(y_cs);

Gamma = diag(sum(y_cs,2)); % Gamma is a diagonal martrix indicating which
                           % nodes are labeled
U = zeros(n,m); % Initializing the membership matrix 
SoS = zeros(n,m); % Initializing the sum of similarities matrix 
Ae=A*ones(n,1);
D = diag(Ae); % The generalized outdegree matrix
L = D-A; % The Laplacian matrix
Dprim = diag(Ae.^(1/2)); % Normalized Laplacian is Dprim*L*Dprim

num = Gamma+lambda*Dprim*L*Dprim;

% Computation of the sum of similarities scores for each class
SoS = (num) \ (Gamma*y_cs);

% Each node is assigned to the class showing the largest sum of
% similarities
[unused,lhat] = max(SoS,[],2); % lhat indicate the class for each n nodes

% Computation of the element of membership matrix U
U = y_cs;
for i=1:n
    if Gamma(i,i) == 0
        U(i,lhat(i)) = 1;
    end
end

end