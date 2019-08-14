function delta_w = oja_rule(w,r_pre,r_post,rate)
% Oja's rule dw/dt = rate*y(t)*(x(t)-y(t)W(t))
delta_w = zeros(size(w,1),size(w,2));
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
    delta_w(i,j) = rate*r_post(i)*(r_pre(j)-r_post(i)*w(i,j));
    end
end
end