function delta_w = bcm_rule(w,performance,r_pre,r_post,rate,r_mean,x_mean)
% Learning rule from Lowenstein and Seung 2006
delta_w = zeros(size(w,1),size(w,2));
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
%     delta_w(i,j) = rate*performance*r_pre(j).*(r_post(i)-...
%                 x_mean(i));
    delta_w(i,j) = rate*performance*(r_pre(j)-...
                r_mean(j)).*(r_post(i)-...
                x_mean(i));
    end
end
end