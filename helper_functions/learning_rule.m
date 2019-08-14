function delta_w = learning_rule(w,performance,r_pre,r_post,rate)
% Learning rule from Lowenstein and Seung 2006
delta_w = zeros(size(w,1),size(w,2));
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
    delta_w(i,j) = rate*performance*r_pre(j).*r_post(i);
    end
end
end