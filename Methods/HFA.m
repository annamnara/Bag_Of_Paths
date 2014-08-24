function [U,SoS] = HFA(A,y_cs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [U,SoS] = HFA(A,y_cl)
% Algorithm 2 The harmonic function approach for labeling the nodes of a
% weigthed directed graph
%
% INPUT ARGUMENTS:
%  A:       nxn matrix, weighted undirected graph G containing n nodes.
%           represented by its symmetric adjacency matrix A.
%  y_cs:    nxm matrix, m binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%
% OUTPUT ARGUMENTS:
%  U:       nxm matrix, membership matrix containing the membership of 
%           each node i to class k, u_ik.
%  SoS:     nxm matrix, indicate the sum of scores of each node i to 
%           class k, sos_ik.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n,m] = size(y_cs);

Gamma_v = sum(y_cs,2); % Gamma_v is a vector indicating which nodes are 
                       % labeled
                          
[unused,index] = sort(Gamma_v,'descend'); % Index is use to sort the nodes
A = A(index,index);                       % All labeled nodes are placed
y_cs = y_cs(index,:);                     % first, then all the unlabeled 
                                          % ones (..._Sorted not indicated
                                          % for visibility)
                          
l = sum(Gamma_v); % l is the (scalar) number of labeled nodes

D = diag(A*ones(n,1)); % The generalized outdegree matrix
P = D\A; % The matrix of transition probabilities (Markov chain)
U_u = zeros(n-l,m); % Initializing the membership matrix only for 
                    % unlabeled nodes
y_u_hat = zeros(n-l,m); % Initializing the sum of similarities matrix only
                      % for unlabeled nodes
P_ul = P(l+1:n,  1:l); % Extract the rows corresponding to unlabeled nodes 
                       % and the columns corresponding to labeled nodes 
                       % from the transition matrix
P_uu = P(l+1:n,l+1:n); % Extract the rows and the columns corresponding to 
                       % unlabeled nodes from the transition matrix
I = eye(length(P_uu)); % Identity matrix (same size than P_uu)

% compute the score for each class c
for c=1:m
    y_u_hat(:,c) = (I-P_uu) \ (P_ul*y_cs(1:l,c));
end

% Each node is assigned to the class showing the highest score
[unused,lhat_u] = max(y_u_hat,[],2); % lhat_u indicate the class for each
                                     % n-l unlabeled nodes

% Computation of the element of membership matrix U
U = y_cs;
for i=l+1:n
    U(i,lhat_u(i-l)) = 1; 
end

SoS = [y_cs ; y_u_hat];                  % U and SoS are ordered back in 
[unused,Reciprocal_index] = sort(index); % the same order than y_cs at the
U = U(Reciprocal_index,:);               % begin. Useful for
SoS = SoS(Reciprocal_index,:);           % cross-validation

end