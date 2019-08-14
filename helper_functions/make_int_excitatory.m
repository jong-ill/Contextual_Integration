function [W_rec,V,D] = make_int_excitatory(dims)
% make_int_excitatory     Make an integrating excitatory network.
%   [W_rec,V,D] = make_int_excitatory(dims) produces an N by N weight 
%   matrix for an excitatory network with eigenvectors V and eigenvalues D.
%   
%   Jon Gill 2018

if nargin < 1 || isempty(dims)
    disp('Setting network dimensions to default n = 100')
    dims = 100;
end

W_rec = rand(dims,dims);
    for i = 1:dims
        W_rec(:,i) = W_rec(:,i)/sum(W_rec(:,i));
    end
    [V,D] = eig(W_rec);
end