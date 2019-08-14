function delta_w = hebbianrule_weightdecay(w,performance,r_pre,r_post,rate,...
                                            alpha)
delta_w = zeros(size(w,1),size(w,2));
% Hebbian learning rule with weight decay.
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
%     delta_w(i,j) = rate*performance*(r_pre(j)-w(i,j)*alpha).*r_post(i);
    delta_w(i,j) = rate*(r_pre(j)-w(i,j)*alpha).*r_post(i);
    end
end
end