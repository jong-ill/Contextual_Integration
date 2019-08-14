function delta_w = hebbian_rule(w,r_pre,r_post,rate)
% Simple hebbian rule 
delta_w = zeros(size(w,1),size(w,2));
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
    delta_w(i,j) = rate*r_post(i)*r_pre(j);
    end
end
end