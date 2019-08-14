function eig_1 = get_int_eig(V,D)

[D_sort,idx] = sort(diag(D),'descend');
V_sort = V(:,idx);
eig_1 = V_sort(:,1);
end