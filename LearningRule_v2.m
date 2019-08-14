% Learning Rule

% Implement hebbian, reward based learning rule. Try to stop before fully
% trained to demonstrate advantage of higher dimensions.

clear all
close all

%% Make Integrating network
single = true;

% Number of Neurons In each population
dims = 100;

if single
    % Single Integrating Network Case
    
    % Integrating Population, recurrent weights
    W_rec = abs(randn(dims,dims)*(1/sqrt(dims))); % Added 1/srt(dims)
    
%     sum(W_rec./sum(W_rec,2),2);
    for i = 1:dims
        W_rec(:,i) = W_rec(:,i)/sum(W_rec(:,i));
    end
    [V,D] = eig(W_rec);
    
    %     % Try normalizing inputs to each cell to 1.
    %     for j = 1:size(W_rec,1)
    %          W_rec(j,:) = W_rec(j,:)/norm(W_rec(j,:));
    %     end
    %
    %     [V,D] = eig(W_rec);
    
else
    % Two competing networks encoding leftward or rightward choices
    
    % Leftward choice network
    W_rec_L = rand(dims,dims);
    for i = 1:dims
        W_rec_L(:,i) = W_rec_L(:,i)/sum(W_rec_L(:,i));
    end
    [V_L,D_L] = eig(W_rec_L);
    
    % Rightward choice network
    W_rec_R = rand(dims,dims);
    for i = 1:dims
        W_rec_R(:,i) = W_rec_R(:,i)/sum(W_rec_R(:,i));
    end
    [V_R,D_R] = eig(W_rec_R);
    
end
%% Check the maximum eigenvalue is 1
if ~single
    plot_eig(D_L)
    plot_eig(D_R)
else
    plot_eig(D)
end
% Get the eigenvector along the integrating dimension
if ~single
    eig_1_L = get_int_eig(V_L,D_L);
    eig_1_R = get_int_eig(V_R,D_R);
else
    eig_1 = get_int_eig(V,D);
end

%% Define sensory layer rates

% Size of input layer
dims_input = 100;

% Initialize weights for stimuli and context (just using stimuli for now)
alpha_c = randn(dims_input,1);
alpha_m = randn(dims_input,1);
alpha_ctx1 = rand(dims_input,1)-0.5;
alpha_ctx2 = rand(dims_input,1)-0.5;
% a_0 = randn(dims_input,1)+4;
a_0 = zeros(dims_input,1);

lambda = 0.001;
lambda = sqrt(lambda);

% Define conditions
context = [1,2];
ctx1 = [1,0];
ctx2 = [0,1];

coherence_values = [0.1 -0.1];
motion_coherence_values = coherence_values;
color_coherence_values = coherence_values;

% Initialize design matrix
numconds = length(ctx1)*length(coherence_values)*2;
R = zeros(dims_input,numconds);
Y = zeros(numconds,dims);
lambda_mat = eye(numconds,numconds)*lambda;

% Build design matrix
for output = 1:dims
    n = 1; % Lets you index across conditions
    for t = 1:length(context)
        for j = 1:length(motion_coherence_values)
            for k = 1:length(color_coherence_values)
                
                % Iterate through condition combinations
                motion = motion_coherence_values(j);
                color = color_coherence_values(k);
                ctx = context(t);
                
                % Get rates of input neurons for all stimulus combinations
                R(:,n) = input_rate(alpha_c,color,alpha_m,motion,...
                    alpha_ctx1,alpha_ctx2,ctx,a_0);
                
                % Define desired response for a given neuron (stimulus sign
                % * context gate).
                Y(n,output) = ((color*ctx1(ctx))+(motion*ctx2(ctx)));
                
                % Multiply by component of integrating eigenvector
                Y(n,output) = Y(n,output)*eig_1(output);
                n = n+1;
            end
        end
    end
end

%% Regress to get the optimal output weights
% methods are 'least squares', 'ridge', 'lasso', etc.

method = 'ridge';

switch lower(method)
    
    % Least squares regression
    case 'least squares'
        
        % Regress, for all input neurons to each output
        for output = 1:dims
            W(:,output) = regress(Y(:,output),R(:,:)');
        end
        
    % Ridge regression
    case 'ridge'
        % Define the ridge paramter (for example, k = 0:1e-5:5e-3;)
%         k = 5e-3;
%         k = 0:1e-5:5e-3;
        % Z-score your predictors (input rates) (note: unnecessary because
        % 'ridge' handles this for you. Left in for other reasons. 
%         inputRatesZscored = zscore(R');
        
        
        
        % Regress, for all input neurons to each output
        k = 5e-3; % define ridge parameter (weight penalization)
        for output = 1:dims
            W(:,output) = ridge(Y(:,output),R(:,:)',k);
        end
        
%         % Regress, sweep across k for plotting
%         k = 0:1e-3:1e-0;
%         W_plot = zeros(dims,dims,length(k));
%         for output = 1:dims
%             W_plot(:,output,:) = ridge(Y(:,output),R(:,:)',k);
%         end
end

% %% Check out weights dependent on ridge parameter
% 
% % Create a ridge plot
% figure
% for i = 1:size(W_plot,1)
%     for j = 1:size(W_plot,2)
%         W_vals = W_plot(i,j,:); % set up a temporary vector
%         W_vals = squeeze(W_vals); % modify so we can transpose
%         plot(k,W_vals','LineWidth',2)
%         hold on
%     end
% end
% ylim([-1 1])
% grid on
% xlabel('Ridge Parameter')
% ylabel('Standardized Coefficient')
% title('{\bf Ridge Trace}')
% 
% % Scatter plot of least squares coefficients vs. highest ridge parameter
% figure;
% ax=gca;
% scatter(repmat(0,size(W_plot,1)*size(W_plot,2),1),reshape(W_plot(:,:,1),size(W_plot,1)*size(W_plot,2),1))
% hold on
% scatter(repmat(1,size(W_plot,1)*size(W_plot,2),1),reshape(W_plot(:,:,end),size(W_plot,1)*size(W_plot,2),1))
% axis([-0.5 1.5 -1 1])
% ylabel('coefficients')
% xlabel('lambda range (1=max)')
% makeNiceFigure(ax)
% 
% % Plot a histogram of the coefficients
% figure
% ax=gca;
% histogram(W_plot(:,:,1),'BinWidth',.05)
% hold on
% histogram(W_plot(:,:,end),'BinWidth',.05)
% axis([-0.5 0.5 0 numel(W)/2])
% legend('least squares','ridge')
% xlabel('coeff.')
% ylabel('count')
% makeNiceFigure(ax)


%% Check Performance
SSE = sum(sum((Y-(R'*W)).^2,1));
if SSE<eps
    disp('SSE tiny, regression converged')
else
    SSE
end

%% Initialize sensory layer-> integrating population
W = W';
%% Initialize with correct answer
W_norm = [];
W_norm = W;



% Ensure that all ouput neurons recieve input on a weight vector with norm
% = 1. Make sure all neurons are generally getting the same amount of
% input.
for j = 1:size(W_norm,1)
    W_norm(j,:) = (W_norm(j,:))/norm(W_norm(j,:));
end

% Calculate the degree of rotation achieved by the input matrix
degree_regress = acosd(eig_1'*W(:,1)/norm(W(:,1)));
disp(['Solution found ' num2str(degree_regress) ' degree rotation'])

%% Randomly add deviations to weight matrix
W_init = randn(dims,dims_input)*.1;

% % Randomly add deviations to weight matrix
% W_init = W_init+randn(dims,dims_input)*1/sqrt(dims);

% % Randomly initialize weights of input layer-> integrating population
% W_init = randn(dims,dims_input);

% Ensure that all ouput neurons recieve input on a weight vector with norm
% = 1. Make sure all neurons are generally getting the same amount of
% input.
for j = 1:size(W_init,1)
    W_init(j,:) = (W_init(j,:))/norm(W_init(j,:));
end

% Calculate the degree of rotation achieved by the input matrix
degree_init = acosd(eig_1'*W_init(:,1)/norm(W_init(:,1)));
disp(['Initialized to ' num2str(degree_init) ' degree rotation'])

% W_train begins as the initial weights and is trained by learning
W_train = W_init;

%% Set up training batch

% Make a look-up table for trial order, randomized to improve learning both
% contexts simultaneously.
context = [1 2];
coherence_values = [-0.15:0.02:0.15];
% coherence_values = [-0.05 0.05];
motion_coherence_values = coherence_values;
color_coherence_values = coherence_values;

correct_ans = [];
trial_conds = [];

trials = 500;

for trial = 1:trials
    idx_m = randperm(length(motion_coherence_values));
    idx_c = randperm(length(color_coherence_values));
    idx_ctx = randperm(length(context));
    
    % Define the conditions for each trial, [context, color, motion]
    trial_conds(trial,:) = [idx_ctx(1) idx_c(1) idx_m(1)];
    
    stim_sign = [sign(color_coherence_values(idx_c(1)))...
        sign(motion_coherence_values(idx_m(1)))];
    
    % Define the correct answer for each trial
    correct_ans(trial) = stim_sign(idx_ctx(1));
end

%% Define Temporal parameters

% Integration time
interval = 1000;

% Biological time constant
t_bio = 10;
integration_dt = t_bio / 100; %integration timestep

% Timesteps for plotting
timesteps = 1:interval;

%% Test the Network

%%% Define training parameters %%%
% Define noise to apply to the sensory signals
noise_std = 0;

% Initialize performance variables (for speed)
choice = zeros(1,trials);
performance = zeros(1,trials);
SSE = zeros(1,trials);
performance_vec = [];
y = zeros(trials,interval);

%%% Run the training %%%

for trial = 1:trials
    % Update of progress
    clc
    disp(['iteration number ' num2str(trial)])
    
    % Initialize dynamic parameters before each trial
    r = zeros(dims_input,interval);
    dx_dt = zeros(dims,interval);
    x = zeros(dims,interval);
    
    % Run the trial
    for t = 1:interval
        
        context = trial_conds(trial,1);
        % Stimulus Period is 500ms in the middle of trial
        if t>250
            % Stimuli + noise
            color = color_coherence_values(trial_conds(trial,2))...
                + randn(1)*noise_std;
            motion = motion_coherence_values(trial_conds(trial,3))...
                + randn(1)*noise_std;
            %             motion = 0;
            
        else
            color = 0;
            motion = 0;
        end
        
        % Rates of input layer
        r(:,t) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
            alpha_ctx2,context,a_0);
        
        % Change of rates in integrator network
        dx_dt(:,t)= (-x(:,t) + W_rec*x(:,t) + W_init*r(:,t))/t_bio;
        
        % Update rates
        x(:,t+1) = x(:,t)+dx_dt(:,t)*integration_dt;
        
        % Readout of integrator
        y(trial,t) = eig_1'*x(:,t);
    end
    
    % Simpler Estimate of performance
    if sign(mean(y(trial,interval-200:interval))) >0
        choice(trial) = 1; %Choose right
    elseif sign(mean(y(trial,interval-200:interval))) <0
        choice(trial) = -1; %Choose left (didn't choose right)
    else
        choice(trial) = 0;
    end
    
    % Compute trial performance
    if choice(trial) == correct_ans(trial)
        performance(trial) = 1*mean(y(trial,interval-200:interval));
    elseif choice(trial) == 0
        performance(trial) = 0;
    else
        performance(trial) = -1*mean(y(trial,interval-200:interval));
    end
    
    performance_vec(trial) = choice(trial)*correct_ans(trial);
    
    % Pin performance value between 1 and -1 to reduce blow up.
    % TODO: This is good, but also need to have saturating rates and 
    % weights. Consider
    % normalization, though optimizing for both tasks should lead to
    % competition. 
    if abs(performance(trial))>1
        performance(trial) = sign(performance(trial));
    end
    
    SSE(trial) = sum(sum((Y-(R'*W_train')).^2,1));
    propCorrect = numel(find(performance_vec==1))/length(performance_vec);
    %     random_proj(n) = eig_1'*M_weights;
    disp(['proportion correct = ' num2str(propCorrect)])
end

propCorrect = numel(find(performance_vec==1))/length(performance_vec);
%     random_proj(n) = eig_1'*M_weights;
disp(['proportion correct = ' num2str(propCorrect)])

% Put everything together for visualization
performance_log = [choice' correct_ans' performance_vec'];


performance_vec(find(performance_vec==-1)) = 0;
figure
plot(movmean(performance_vec,50))

%% Plot the integration
figure
ax = gca;
for trial = 1:trials
    context = trial_conds(trial,1);
    % Plot the integration
    if context == 1 && correct_ans(trial) == 1
        plot(y(trial,:),'r','LineWidth',2)
    elseif context == 2 && correct_ans(trial) == 1
        plot(y(trial,:),'m','LineWidth',2)
    elseif context == 1 && correct_ans(trial) == -1
        plot(y(trial,:),'g','LineWidth',2)
    elseif context == 2 && correct_ans(trial) == -1
        plot(y(trial,:),'c','LineWidth',2)
    end
    hold on
end
xlabel('timesteps')
ylabel('Integrated value')
axis([0 interval -1 1])
plot(1:1000,repmat(0,1000))
makeNiceFigure(ax)


%% Run Training

%%% Define training parameters %%%
% Define noise to apply to the sensory signals
noise_std = 1;

% Initialize performance variables (for speed)
choice = zeros(1,trials);
performance = zeros(1,trials);
SSE = zeros(1,trials);
performance_vec = [];
propCorrect = 0;
x_session = zeros(dims,trials);
r_session = zeros(dims_input,trials);
r_sum = 0;
r_mean = 0;
x_sum = 0;
x_mean = 0;
y = zeros(trials,interval);


% Initialize learning parameters
rate = 0.1;

rho = 0.1;

%%% Run the training %%%

for trial = 1:trials
    % Update of progress
    clc
    disp(['iteration number ' num2str(trial)])
    disp(['proportion correct = ' num2str(propCorrect)])
    
    % Initialize dynamic parameters before each trial
    r = zeros(dims_input,interval);
    dx_dt = zeros(dims,interval);
    x = zeros(dims,interval);
    
    % Run the trial
    for t = 1:interval
        
        context = trial_conds(trial,1);
        % Stimulus Period is 500ms in the middle of trial
        if t>250
            % Stimuli + noise
            color = color_coherence_values(trial_conds(trial,2))...
                + randn(1)*noise_std;
            motion = motion_coherence_values(trial_conds(trial,3))...
                + randn(1)*noise_std;
            %             motion = 0;
            
        else
            color = 0;
            motion = 0;
        end
        
        % Rates of input layer
        r(:,t) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
            alpha_ctx2,context,a_0);
        
        % Change of rates in integrator network
        dx_dt(:,t)= (-x(:,t) + W_rec*x(:,t) + W_train*r(:,t)+...
                                    rho*randn(size(x,1),1))/t_bio;
        
        % Update rates
        x(:,t+1) = x(:,t)+dx_dt(:,t)*integration_dt;
        
        % Readout of integrator
        y(trial,t) = eig_1'*x(:,t);
    end
    
%     % Simpler Estimate of performance
    if sign(mean(y(trial,interval-200:interval))) >0
        choice(trial) = 1; %Choose right
    elseif sign(mean(y(trial,interval-200:interval))) <0
        choice(trial) = -1; %Choose left (didn't choose right)
    else
        choice(trial) = 0;
    end
    
    % Estimate performance using thresholds
%     if mean(y(trial,interval-200:interval)) >=0.1
%         choice(trial) = 1; %Choose right
%     elseif mean(y(trial,interval-200:interval)) <=-0.1
%         choice(trial) = -1; %Choose left (didn't choose right)
%     else
%         choice(trial) = 0;
%     end
    
    % Compute trial performance
    if choice(trial) == correct_ans(trial)
        performance(trial) = 1*mean(y(trial,interval-200:interval));
    elseif choice(trial) == 0
        performance(trial) = 0;
    else
        performance(trial) = -1*mean(y(trial,interval-200:interval));
    end
    
    performance_vec(trial) = choice(trial)*correct_ans(trial);
    
    % Pin performance value between 1 and -1 to reduce blow up.
    % TODO: This is good, but also need to have saturating rates and 
    % weights. Consider
    % normalization, though optimizing for both tasks should lead to
    % competition. 
    if abs(performance(trial))>1
        performance(trial) = sign(performance(trial));
    end
    
  
    
    r_pre = mean(r(:,interval-200:interval),2);
    r_post = mean(x(:,interval-200:interval),2);
    
%     x_session(:,trial) = r_post;
%     r_session(:,trial) = r_pre;
    
    x_sum = x_sum+r_post;
    x_mean = x_sum/trial;
    
    r_sum = r_sum+r_pre;
    r_mean = r_sum/trial;
    
    %Update Weights
    
%     alpha = 0.1;
    delta_w = learning_rule(W_train,performance(trial),r_pre,r_post,rate);

%       delta_w = hebbianrule_weightdecay(W_train,performance(trial),r_pre,r_post,rate,...
%                                             alpha);

%         delta_w = hebbian_rule(W_train,r_pre,r_post,rate);
%      delta_w = bcm_rule(W_rec,performance(trial),r_pre,r_post,rate,r_mean,x_mean);
%     
%     delta_w = oja_rule(W_train, mean(r(:,interval-200:interval),2), ...
%         mean(x(:,interval-200:interval),2), 0.01);
%      delta_w = EH_rule(W_rec,performance(trial),r_pre,r_post,rate);

    
    W_train = W_train+delta_w;
    % Normalize so every unit in the RNN gets inputs of norm = 1
    
    for j = 1:size(W_train,1)
         W_train(j,:) = W_train(j,:)/norm(W_train(j,:));
    end
%     
%     W_train = W_train+delta_w;
    
    SSE(trial) = sum(sum((Y-(R'*W_train')).^2,1));
    propCorrect = numel(find(performance_vec==1))/length(performance_vec);
    %     random_proj(n) = eig_1'*M_weights;
   
end
performance_vec(find(performance_vec==-1)) = 0;
figure
plot(movmean(performance_vec,50))

% Put everything together for visualization
performance_log = [choice' correct_ans' performance_vec'];

% Evaluate the degrees of rotation towards the integrating mode. 
% for j = 1:size(W_train,1)
%     W_norm(j,:) = (W_train(j,:))/norm(W_train(j,:));
% end

% Calculate the degree of rotation achieved by the input matrix
degree_train = acosd(eig_1'*W_train(:,2)/norm(W_train(:,2)));
disp(['Solution found ' num2str(degree_train) ' degree rotation'])

%% Plot Performance over training

figure
ax = gca;
plot([1:trials],SSE)
xlabel('Iteration')
ylabel('Error (SSE)')
makeNiceFigure(ax)
title('training performance')

figure
ax=gca;
plot(performance)
makeNiceFigure(ax)
title('performance')

%% Check out weight matricies

figure
subplot(1,3,1)
imagesc(W)
colorbar
subplot(1,3,2)
imagesc(W_init)
colorbar
subplot(1,3,3)
imagesc(W_train)
colorbar

%% Test the network

% Test trials
trials = numconds*10;

% Make a look-up table for trial order, randomized to improve learning both
% contexts simultaneously.
context = [1 2];
coherence_values = [0.1 -0.1];
motion_coherence_values = coherence_values;
color_coherence_values = coherence_values;

for trial = 1:trials
    idx_m = randperm(length(motion_coherence_values));
    idx_c = randperm(length(color_coherence_values));
    idx_ctx = randperm(length(context));
    
    % Define the conditions for each trial, [context, color, motion]
    trial_conds(trial,:) = [idx_ctx(1) idx_c(1) idx_m(1)];
    
    stim_sign = [sign(color_coherence_values(idx_c(1)))...
        sign(motion_coherence_values(idx_m(1)))];
    
    % Define the correct answer for each trial
    correct_ans(trial) = stim_sign(idx_ctx(1));
end

% Define noise to apply to the sensory signals
noise_std = 0.1;

% Initialize rates for network
x = zeros(dims,1);


% Initialize performance variables (for speed)
choice = zeros(1,trials);
performance = zeros(1,trials);
SSE = zeros(1,trials);

for trial = 1:trials
    % Update of progress
    clc
    disp(['iteration number ' num2str(trial)])
    
    % Initialize dynamic parameters before each trial
    r = zeros(dims_input,interval);
    dx_dt = zeros(dims,interval);
    y = zeros(1,interval);
    
    % Run the trial
    for t = 1:interval
        
        context = trial_conds(trial,1);
        % Stimulus Period is 500ms in the middle of trial
        if t>250
            % Stimuli + noise
            color = color_coherence_values(trial_conds(trial,2))...
                + randn(1)*noise_std;
            motion = motion_coherence_values(trial_conds(trial,3))...
                + randn(1)*noise_std;
            %             motion = 0;
            
        else
            color = 0;
            motion = 0;
        end
        
        % Rates of input layer
        r(:,t) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
            alpha_ctx2,context,a_0);
        
        % Change of rates in integrator network
        dx_dt(:,t)= (-x(:,t) + W_rec*x(:,t) + W_train*r(:,t))/t_bio;
        
        % Update rates
        x(:,t+1) = x(:,t)+dx_dt(:,t)*integration_dt;
        
        % Readout of integrator
        y(t) = eig_1'*x(:,t);
    end
    
    
    % Simpler Estimate of performance
    if sign(mean(y(interval-200:interval))) >=0
        choice(trial) = 1; %Choose right
    elseif sign(mean(y(interval-200:interval))) < 0
        choice(trial) = -1; %Choose left (didn't choose right)
    end
    
    % Compute trial performance
    if choice(trial) == correct_ans(trial)
        performance(trial) = 1*mean(y(interval-200:interval));
    else
        performance(trial) = -1*mean(y(interval-200:interval));
    end
    
    % Pin performance value between 1 and -1 to reduce blow up.
    % TODO: This is good, but also need to have saturating rates and 
    % weights. Consider
    % normalization, though optimizing for both tasks should lead to
    % competition. 
    if abs(performance(trial))>1
        performance(trial) = sign(performance(trial));
    end
    
end

% Plot the integration
figure
ax = gca;
for trial = 1:trials
    context = trial_conds(trial,1);
    % Plot the integration
    if context == 1 && correct_ans(trial) == 1
        plot(y,'r','LineWidth',2)
    elseif context == 2 && correct_ans(trial) == 1
        plot(y,'m','LineWidth',2)
    elseif context == 1 && correct_ans(trial) == -1
        plot(y,'g','LineWidth',2)
    elseif context == 2 && correct_ans(trial) == -1
        plot(y,'c','LineWidth',2)
    end
    hold on
end
xlabel('timesteps')
ylabel('Integrated value')
axis([0 interval -1 1])
plot(1:1000,repmat(0,1000))
makeNiceFigure(ax)
