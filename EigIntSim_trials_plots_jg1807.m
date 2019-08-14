clear all
close all

%% Define parameters

% Integration time
interval = 1000;

% Number of Neurons In each population
dims = 10;

% Stimulus Characteristics
coherence_values = [0.009, 0.036, 0.15];
M_weights = (rand(dims,1)*2)-1; %Uniform weights
% M_weights = (randn(dims,1))*(1/dims); % Gaussian weights, 0=mean, std=1/N
M_weights = M_weights/norm(M_weights);

% Initial activation

% Biological time constant
t_bio = 10;
integration_dt = t_bio / 100; %integration timestep


% Input vector (proof of principle exp)
% r = (rand(dims,1)*2)-1;
% r = r/norm(r);
% x = rand(dims,1);
x = zeros(dims,1);

% Integrating Population, recurrent weights
W_rec = rand(dims,dims);
for i = 1:dims
    W_rec(:,i) = W_rec(:,i)/sum(W_rec(:,i));
end
[V,D] = eig(W_rec);

%% Check the maximum eigenvalue is 1
figure()
plot(real(diag(D)),imag(diag(D)),'r*') %   Plot real and imaginary parts
xlabel('Real')
ylabel('Imaginary')
t1 = ['Eigenvalues of a random matrix of dimension ' num2str(dims)];
title(t1)

% Get the eigenvector along the integrating dimension
[D_sort,idx] = sort(diag(D),'descend');
V_sort = V(:,idx);
eig_1 = V_sort(:,1);

%% Run simulations with random or aligned input weights
figure
max_sims = 100;
coherence_values = [0.009, 0.036, 0.15];

for n = 1:max_sims
    
    M_weights = (rand(dims,1)*2)-1;
    M_weights = M_weights/norm(M_weights);
    %
    % if mod(n,max_sims)==0
    %     M_weights = eig_1;
    % end
    
    for i = 1:interval
        %     dx_dt(:,t)= -x(:,t) + W_rec*x(:,t) + M_weights*(coherence_values(3)+randn(1));
        dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + M_weights*coherence_values(3))/t_bio;
        x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
        y(i) = eig_1'*x(:,i);
    end
    
    plot(y,'k','LineWidth',2)
    hold on
    xlabel('timesteps')
    ylabel('Integrated value')
    
    random_proj(n) = eig_1'*M_weights;
end

% Plot integration of input weights aligned with first eigenvector
M_weights = eig_1;

for i = 1:interval
    %     dx_dt(:,t)= -x(:,t) + W_rec*x(:,t) + M_weights*(coherence_values(3)+randn(1));
    dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + M_weights*coherence_values(3))/t_bio;
    x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
    y(i) = eig_1'*x(:,i);
end
timesteps = 1:interval;
plot(timesteps,y,'r-','LineWidth',3)
% Try ode45

%% Determine how STD of random projections scales with dimensionality
figure
dim_vec = 1:1000000;
proj_std = 1./sqrt(dim_vec);
% semilogx(dim_vec,rad2deg(proj_std),'k','LineWidth','3')
semilogx(dim_vec,rad2deg(proj_std),'k','LineWidth',5)

xlabel('Dimensions')
ylabel('Std of projection (deg)')
set(gca,'fontsize',30)
set(gca,'box','off')
% 
% figure

hold on

% dim_vec = 1:50000;
proj_std = 1./sqrt(dim_vec);
% semilogx(dim_vec,rad2deg(proj_std),'k','LineWidth','3')
semilogx(dim_vec,rad2deg(proj_std)*10,'r--','LineWidth',5)

xlabel('Dimensions')
ylabel('Std of projection (deg)')
axis([dim_vec(1) dim_vec(end) 0 90])
set(gca,'fontsize',30)
set(gca,'box','off')
yticks(10:20:90)
xticks([1e0  1e2  1e4  1e6])



% Plot angle necessary to be 10x over the noise
figure
semilogx(dim_vec,rad2deg(proj_std)*2)

figure
semilogx(dim_vec,exp((-1.*dim_vec).*((proj_std*10).^2)))

% Plot mean
figure
dim_vec = 1:1000:50000;

for j = 1:length(dim_vec)
    dims = dim_vec(j);
    for i = 1:1000
        M_weights = (rand(dims,1)*2)-1;
        M_weights = M_weights/norm(M_weights);
        C_weights = (rand(dims,1)*2)-1;
        C_weights = C_weights/norm(C_weights);
        dp(i,j) = M_weights'*C_weights;
    end
end

semilogx(dim_vec,rad2deg(mean(dp,1)),'k','LineWidth',5)
xlabel('dimensions')
ylabel('mean of projection')
axis([dim_vec(1) dim_vec(end) -180 180])
set(gca,'fontsize',30)
set(gca,'box','off')




%% Add a second input:
coherence_values = [0.009, 0.036, 0.15];
C_weights = (rand(dims,1)*2)-1;
C_weights = C_weights/norm(C_weights);

figure
max_sims = 100;
for n = 1:max_sims
    
    M_weights = (rand(dims,1)*2)-1;
    M_weights = M_weights/norm(M_weights);
    %
    % if mod(n,max_sims)==0
    %     M_weights = eig_1;
    % end
    
    for i = 1:interval
        %     dx_dt(:,t)= -x(:,t) + W_rec*x(:,t) + M_weights*(coherence_values(3)+randn(1));
        dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + M_weights*coherence_values(3)...
            +C_weights*coherence_values(3))/t_bio;
        x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
        y(i) = eig_1'*x(:,i);
    end
    
    plot(y)
    hold on
    xlabel('timesteps')
    ylabel('Integrated value')
    
    random_proj(n) = eig_1'*M_weights;
end

% Plot integration of input weights aligned with first eigenvector
C_weights = -eig_1;

for i = 1:interval
    %     dx_dt(:,t)= -x(:,t) + W_rec*x(:,t) + M_weights*(coherence_values(3)+randn(1));
    dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + M_weights*coherence_values(3)...
        +C_weights*coherence_values(3))/t_bio;
    x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
    y(i) = eig_1'*x(:,i);
end
timesteps = 1:interval;
plot(timesteps,y,'r-','LineWidth',2)

%% Simulate the Contextual Modulation
% Stimulus Characteristics
coherence_values = [0.009, 0.036, 0.15 -0.009, -0.036, -0.15];
M_weights = (rand(dims,1)*2)-1;
M_weights = M_weights/norm(M_weights);
C_weights = (rand(dims,1)*2)-1;
C_weights = C_weights/norm(C_weights);
CC_weights = eig_1;
CM_weights = eig_1;

attendmotion = 0;

if attendmotion
    u_cc = 0;
    u_cm = 1;
    attend = 'Motion';
else
    u_cc = 1;
    u_cm = 0;
    attend = 'Color';
end

I = eye(dims);

for i = 1:interval
    dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + I*(M_weights*(coherence_values(3)+rand(1))+CM_weights*u_cm*coherence_values(3))...
        +I*(C_weights*(coherence_values(6)+rand(1))+CC_weights*u_cc*coherence_values(6)))/t_bio;
    
    %         dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + I*(M_weights*coherence_values(3)+CM_weights*u_cm*coherence_values(3))...
    %             +I*(C_weights*coherence_values(6)+CC_weights*u_cc*coherence_values(6)))/t_bio;
    x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
    y(i) = eig_1'*x(:,i);
end

plot(y)
hold on
xlabel('timesteps')
ylabel('Integrated value')
title(['Attend to ' attend])
%% Clear Previous runs
clear all
close all
%% Simulate the Contextual Modulation w/ populations of different sizes

% Integration time
interval = 1000;

% Number of Trials
trials = 10;

% Number of Neurons In each population
dims_in = 100;
dims_out = 100;

% Gain of noise in stimulus
noise_gain = 0;

% Attend Condition
% context = 1; % 1 or -1
% 
% if context == 1
%     u_cc = 0;
%     u_cm = 1;
%     attend = 'Motion';
% elseif context == -1
%     u_cc = 1;
%     u_cm = 0;
%     attend = 'Color';
% elseif context == 0
%     u_cc = 0;
%     u_cm = 0;
%     attend = 'No Cue';
%     
% end

% Biological time constant
t_bio = 10;
integration_dt = t_bio / 100; %integration timestep

% Integrating Population, recurrent weights
W_rec = make_integrator(dims_out, 1e-2);


[V,D] = eig(W_rec);
[D_sort,idx] = sort(diag(D),'descend');
V_sort = V(:,idx);
eig_1 = V_sort(:,1);

% Stimulus Characteristics
% coherence_values = [0.009, 0.036, 0.15 -0.009, -0.036, -0.15];
coherence_values = [0.15, -0.15];

% motion_coherence = [0.15 -0.15];
% color_coherence = [0.15 -0.15];
motion_coherence = coherence_values;
color_coherence = coherence_values;
% 
% motion_coherence = 0;
% color_coherence = 0;

% M_weights = (rand(dims_in,1)*2)-1;
% M_weights = M_weights/norm(M_weights);
% C_weights = (rand(dims_in,1)*2)-1;
% C_weights = C_weights/norm(C_weights);

M_weights = randn(dims_in,1);
M_weights = M_weights/norm(M_weights);
C_weights = randn(dims_in,1);
C_weights = C_weights/norm(C_weights);

% Initialize Rates
I = eye(dims_in);
x = zeros(dims_out,1);

W_c = randn(dims_out,dims_in)./dims_in;
W_m = randn(dims_out,dims_in)./dims_in;
CC_weights = W_c\eig_1;
CM_weights = W_m\eig_1;


F = [];
x_trials_cat = [];
total_trials = 0;
cond_avg = [];

% for context = [0 -1 1]

context_values = [-1 1];
% for context = [-1 1]
% 
%     
% if context == 1
%     u_cc = 0;
%     u_cm = 1;
%     attend = 'Motion';
%     c_idx = 1;
% elseif context == -1
%     u_cc = 1;
%     u_cm = 0;
%     attend = 'Color';
%     c_idx = 2;
% elseif context == 0
%     u_cc = 0;
%     u_cm = 0;
%     attend = 'No Cue';  
%     c_idx = 3;
% end

design_matrix = [];
for j = 1:length(motion_coherence)
    for k = 1:length(color_coherence)
        for ctx = 1:length(context_values)
            design_matrix(end+1,:) = [j,k,ctx];
         end
     end
end

trialNum = zeros(dims_out,size(design_matrix,1),2);

for cond = 1:size(design_matrix,1)
    j = design_matrix(cond,1);
    k = design_matrix(cond,2);
    c_idx = design_matrix(cond,3);
    
if c_idx == 1
    u_cc = 0;
    u_cm = 1;
    attend = 'Motion';
%     c_idx = 1;
elseif c_idx == 2
    u_cc = 1;
    u_cm = 0;
    attend = 'Color';
%     c_idx = 2;
end


% for j = 1:length(motion_coherence)
% for k = 1:length(color_coherence)
    

for l= 1:trials
    total_trials = total_trials+1;
    stim_on = 1;
    choice = sign(motion_coherence(j)*u_cm + color_coherence(k)*u_cc);
for i = 1:interval
    if i >=100 && i<800
        stim_on = 1;
    else
        stim_on = 0;
    end
    dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + W_m*(M_weights*...
        ((motion_coherence(j)*stim_on)+(randn(1)*noise_gain))+CM_weights*u_cm*...
        motion_coherence(j)*stim_on)+W_c*(C_weights*((color_coherence(k)*stim_on)+...
        randn(1)*noise_gain)+CC_weights*u_cc*color_coherence(k)*stim_on))/t_bio;
if i~=interval
    x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
end
    y(i) = eig_1'*x(:,i);
    
end

choice = sign(mean(y(1:round(end/2))));
if choice>=0
    decision = 1;
else 
    decision = 2;
end
% somehow get 1:8 out of stimulus combinations. 
trialNum(1:dims_out,cond,decision) = trialNum(1:dims_out,cond,decision)+1;
firingRates(1:dims_out,cond,decision,1:interval,l) = x;

% x_trials(:,:,l,k,j,c_idx) = x;
% x_trials_cat(:,:,end+1) = x;
% x_trials_temp(:,:,l) = x;
y_trials(l,:,c_idx) = y;


% F(1,total_trials) = choice;
% F(2,total_trials) = motion_coherence(j);
% F(3,total_trials) = color_coherence(k);
% F(4,total_trials) = context;
% F(5,total_trials) = 1;
end
% cond_avg = cat(2,cond_avg,mean(x_trials_temp,3));

end

firingRatesAverage = nanmean(firingRates,5);

% save('dpca_temp_data.mat','trialNum','firingRates','firingRatesAverage')

%% Plot the result of integration
figure
for ccc = 3
for pp = 1:size(y_trials,1)
plot(y_trials(pp,:,ccc),'k--')
hold on
end
end
xlabel('timesteps')
ylabel('Integrated value')
% title(['Attend to ' attend])

plot(mean(y_trials(:,:,1),1),'k','LineWidth',4)
plot(mean(y_trials(:,:,2),1),'b','LineWidth',4)

set(gca,'fontsize',30)
set(gca,'box','off')
axis([0 1000 -.4 .4])
%% Plot Firing Rates 
figure
for idx=1:size(x,1)
    plot(x(idx,:))
    hold on
end
hold off

%% Analyze activity in recurrent network using PCA and Regression

% Build a big data matrix with mean subtracted average unit responses

% % Get data for each condition across trials
% condition_1(:,:,:) = x_trials(:,:,:,1,1);
% condition_2(:,:,:) = x_trials(:,:,:,2,1);
% condition_3(:,:,:) = x_trials(:,:,:,1,2);
% condition_4(:,:,:) = x_trials(:,:,:,2,2);
% 
% % Average across trials per condition
% cond_1_mean = mean(condition_1,3);
% cond_2_mean = mean(condition_2,3);
% cond_3_mean = mean(condition_3,3);
% cond_4_mean = mean(condition_4,3);
% 
% % Append mean activity for each condition 
% x_mean_append = [cond_1_mean cond_2_mean cond_3_mean cond_4_mean];

% Subtract the mean of each unit across conditions
% x_mean_append = (x_mean_append-mean(x_mean_append,2))./std(x_mean_append,0,2);
x_mean_append = (cond_avg-mean(cond_avg,2))./std(cond_avg,0,2);

% Principal Components Analysis to get de-noising matrix
% [V_a,score,latent,~,explained] = pca(x_mean_append);
[Va, ~] = eig(cov(x_mean_append'));
N_pca = 12;
D = Va(:, 1:N_pca) * Va(:, 1:N_pca)';

%% Regress data on condition axes

% Subtract the mean of each unit across conditions
% x_append = [condition_1 condition_2 condition_3 condition_4];
% x_append = (x_append-mean(x_append,2))./std(x_append,0,2);

x_trials_cat = x_trials_cat(:,:,1:end-1);
% x_trials_cat = (x_trials_cat-mean(mean(x_trials_cat,2),3))./std(std(x_trials_cat,0,2),0,3);

x_trials_append = reshape(x_trials_cat,[dims_out,size(x_trials_cat,2)*size(x_trials_cat,3)]);
grand_mean = mean(x_trials_append,2);
grand_std = std(x_trials_append,0,2);

x_trials_norm = (x_trials_cat-grand_mean)./grand_std;
x_trials_avg_norm = mean(x_trials_norm,3);

% x_trials_append = (x_trials_append-mean(x_trials_append(2)))./std(x_trials_append,0,2);

for unit = 1:dims_out
    for timepoint = 1:interval
        trial_rates(:,1) = x_trials_norm(unit,timepoint,:);
        B(:,timepoint,unit) = inv(F*F')*F*trial_rates;
        trial_rates = [];
    end
end

%% Plot Betas 
figure
for idx=1:size(x,1)
    plot(x(idx,:))
    hold on
end
hold off

%% 
Bpca = zeros(size(B));
for v = 1:size(B, 1)
    Bm = squeeze(B(v,:,:));
    Bpca(v, :, :) = Bm * D;
end
Bn = squeeze(sqrt(sum(Bpca.^2, 3)));
[~,tmax] = max(Bn, [], 2);
Bmax = zeros(size(B, 1), size(B, 3));
for v = 1:size(Bpca, 1)
    t = tmax(v);
    Bmax(v,:) = Bpca(v, t, :);
end
[Q, R] = qr(Bmax(1:4,:)');
Bcperp = Q(:,1:4);

% pvc = Bcperp' * x_mean_append;
pvc = Bcperp' * x_mean_append;


for t = 1:1000:7000
plot(pvc(1, t:t+1000), pvc(2, t:t+1000), '-.')
hold on
end

figure
plot(pvc(1,3001:4000),pvc(3,3001:4000),'-.')
hold on

figure
plot(pvc(4,1:1000))
