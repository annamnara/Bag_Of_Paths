function [U,SoB] = BagOfP(A,y_cs,theta,C)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [U,Gbet] = BagOfP(A,y_cs,theta,C)
% Algorithm 5 The bag of hitting paths group betweenness approach for 
% labeling the nodes of a weighted directed graph G without self-loops.
%
% INPUT ARGUMENTS:.
%  A:       nxn matrix, weighted undirected graph G containing n nodes.
%           represented by its symmetric adjacency matrix A.
%  y_cs:    nxm matrix, m binary indicator vectors y_c containing as 
%           entries 1 for nodes belonging to the class whose label index 
%           is c, and 0 otherwise.
%  theta:   the (scalar) inverse temperature parameter.
%  C:       nxn matrix, cost matrix C associated to G (if not specified,
%           the costs are the inverse of the affinities, but others 
%           choices are possible).
%
% OUTPUT ARGUMENTS:
%  U:       nxm matrix, membership matrix containing the membership of 
%           each node i to class k, u_ik.
%  SoB:     nxm matrix, indicate the sum of group betweennesses of each
%           node i to class k, sob_ik.
%
% (c) 2011-2012 B. Lebichot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin == 3
    C = 1./A; % possibility to call the method with only 3 parameters
end

[n,m] = size(y_cs);


Gamma = diag(sum(y_cs,2)); % Gamma is a diagonal martrix indicating which
                           % nodes are labeled
U = zeros(n,m); % Initializing the membership matrix 
SoB = zeros(n,m); % Initializing the sum of similarities matrix
D = diag(A*ones(n,1)); % The generalized outdegree matrix
P_ref = D\A; % The reference transition probabilities matrix
W = P_ref.*exp(-theta*C); % theta is used to favor some paths
I = eye(size(W)); % Identity matrix (same size than W)
Z = (I-W)\I; % The fundamental matrix
Z0 = Z - diag(diag(Z));
Dz = diag(diag(Z));
Dzinv = diag(diag(Z).^-1);

for c=1:m
    
    hc = y_cs(:,c);
    Gbetc = Dzinv * ((Z0'*hc).*(Z0*hc) - (Z0'.*Z0)*hc); % compute the group
                                                        % betweenness for 
                                                        % class c
    SoB(:,c) = Gbetc/norm(Gbetc); % normalize the betweenness scores
    
end

% Each node is assigned to the class showing the highest group betweenness 
[unused,lhat] = max(SoB,[],2); % lhat indicate the class for each n nodes

% Computation of the element of membership matrix U
U = y_cs;
for i=1:n
    if Gamma(i,i) == 0
        U(i,lhat(i)) = 1;
    end
end

end