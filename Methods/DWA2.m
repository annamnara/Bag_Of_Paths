function [U,SoB] = DWA2(A,y_cs,alpha)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [U,SoS] = DWA2(A,y_cs,alpha)
% Algorithm 4* The D-walk approach for labeling the nodes of a weigthed
% directed graph G without self-loops
%
% INPUT ARGUMENTS:
%  A:       nxn matrix, weighted undirected graph G containing n nodes.
%           represented by its symmetric adjacency matrix A.
%  y_cs:    nxm matrix, m binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%  alpha:   the probability for each step to continue the walk, alpha must
%           be in ]0,1].
%
% OUTPUT ARGUMENTS:
%  U:       nxm matrix, membership matrix containing the membership of 
%           each node i to class k, u_ik.
%  SoB:     nxm matrix, indicate the sum of group betweennesses of each
%           node i to class k, sob_ik.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if alpha < 0 || alpha > 1
    display('error : alpha must be in [0,1]')
end

[n,m] = size(y_cs);

Gamma = diag(sum(y_cs,2)); % Gamma is a diagonal martrix indicating which
                           % nodes are labeled
U = zeros(n,m); % Initializing the membership matrix 
SoB = zeros(n,m); % Initializing the sum of similarities matrix 
D = diag(A*ones(n,1)); % The generalized outdegree matrix
P = alpha.*(D\A); % The matrix of transition probabilities (Markov chain)
I = eye(size(P)); % Identity matrix (same size than P)
n_cs = sum(y_cs); % Compute the number of nodes of each class

% Computation of the group betweenness for each class c
for c=1:m
    Q_c = diag(y_cs(:,c))*P; % Set rows corresponding to class c to zeros
                             % in order to produce absorbing nodes
    SoB(:,c) = (1/n_cs(c)) * ( (I-Q_c') \ (P*y_cs(:,c)) );
end

% Each node is assigned to the class showing highest group betweenness
[unused,lhat] = max(SoB,[],2); % lhat indicate the class for each n nodes

% Computation of the element of membership matrix U
U = y_cs;
for i=1:n
    if Gamma(i,i) == 0
        U(i,lhat(i)) = 1;
    end
end

end