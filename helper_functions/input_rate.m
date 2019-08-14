function r = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,alpha_ctx2,context,a_0)

% Initialize logic of context conditions. 
ctx1 = [1,0];
ctx2 = [0,1];

% Rate is a linear combination of color and motion inputs scaled by
% modulatory input. 
if context>0
    modulation = ((alpha_ctx1*ctx1(context)) + (alpha_ctx2*ctx2(context)));
else
    modulation = 0;
end

r = (((alpha_c*color) + (alpha_m*motion)).*(1+modulation))+a_0; % a_0 not modulated

% Rectify firing rates to make sure you don't go negative (r=max([0,r])
idx = r<0; r(idx) = 0;

end
