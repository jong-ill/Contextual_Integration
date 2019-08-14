function delta_w = EH_rule(w,performance,r_pre,r_post,rate)
% Learning rule from Lowenstein and Seung 2006
% Adapted to compute degree away from target (1) instead of raw integrated
% activity.
performance = (1-performance)*sign(performance);

delta_w = zeros(size(w,1),size(w,2));
for i = 1:size(r_post,1)
    for j = 1:size(r_pre,1)
%     delta_w(i,j) = rate*performance*r_pre(j).*(r_post(i)-...
%                 x_mean(i));
    delta_w(i,j) = rate*performance*r_pre(j)*r_post(i);
    end
end
end