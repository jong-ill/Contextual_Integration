function [W_rec,V,D] = make_int_ei(dims)
% make_int_ei     Make an integrating EI network.
%   [W_rec,V,D] = make_int_ei(dims) produces an N by N weight matrix for an
%   EI network with eigenvectors V and eigenvalues D. Dale's law is not
%   obeyed.
%   
%   Jon Gill 2019

if nargin < 1 || isempty(dims)
    disp('Setting network dimensions to default n = 100')
    dims = 100;
end

W_rec = randn(dims,dims).*(1/dims);
    for i = 1:dims
        W_rec(:,i) = W_rec(:,i)/sum(W_rec(:,i));
    end
    [V,D] = eig(W_rec);
end