% Learning Rule

% Implement hebbian, reward based learning rule. Try to stop before fully
% trained to demonstrate advantage of higher dimensions.

clear all
close all

%% Set up populations and conditions

% Number of trials to train on.
trials = 800;

% Number of neurons in integrating population
dims = 100;

% Integrating Population, recurrent weights
[W_rec,V,D] = make_int_excitatory(dims);

[W_rec,V,D] = make_int_excitatory_norm(dims);

% Define integrating eigenvector
eig_1 = get_int_eig(V,D);

%% Define sensory layer rates

% Size of input layer
dims_input = 10;

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
        k = 0:1e-5:5e-3;
        % Z-score your predictors (input rates) (note: unnecessary because
        % 'ridge' handles this for you. Left in for other reasons. 
%         inputRatesZscored = zscore(R');
        
        
        
        % Regress, for all input neurons to each output
        k = 5e-3; % define ridge parameter (weight penalization)
        for output = 1:dims
            W(:,output) = ridge(Y(:,output),R(:,:)',k);
        end
        
        % Regress, sweep across k for plotting
        k = 0:1e-5:5e-3;
        W_plot = zeros(dims,dims,length(k));
        for output = 1:dims
            W_plot(:,output,:) = ridge(Y(:,output),R(:,:)',k);
        end
end

%% Check Performance
SSE = sum(sum((Y-(R'*W)).^2,1));
if SSE<eps
    disp('SSE tiny, regression converged')
else
    SSE
end

%% Initialize sensory layer-> integrating population

% Initialize with correct answer
W_init = W;

% Randomly initialize weights of input layer-> integrating population
% W_init = randn(dims,dims_input);

% Ensure that all ouput neurons recieve input on a weight vector with norm
% = 1. Make sure all neurons are generally getting the same amount of
% input.
for j = 1:size(W_init,1)
    W_init(j,:) = (W_init(j,:))/norm(W_init(j,:));
end

% W_train begins as the initial weights and is trained by learning
W_train = W_init;

%% Set up training batch

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

%% Define Temporal parameters

% Integration time
interval = 1000;

% Biological time constant
t_bio = 10;
integration_dt = t_bio / 100; %integration timestep

% Timesteps for plotting
timesteps = 1:interval;

%% Run Training

%%% Define training parameters %%%
% Define noise to apply to the sensory signals
noise_std = 0.1;

% Initialize performance variables (for speed)
choice = zeros(1,trials);
performance = zeros(1,trials);
SSE = zeros(1,trials);

% Initialize learning parameters
rate = 0.1;

%%% Run the training %%%

for trial = 1:trials
    % Update of progress
    clc
    disp(['iteration number ' num2str(trial)])
    
    % Initialize dynamic parameters before each trial
    r = zeros(dims_input,interval);
    dx_dt = zeros(dims,interval);
    y = zeros(1,interval);
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
    
    r_pre = mean(r(:,interval-200:interval),2);
    r_post = mean(x(:,interval-200:interval),2);
    
    %Update Weights
    delta_w = learning_rule(W_train,performance(trial)*.1,r_pre,r_post,rate);

%       delta_w = hebbianrule_weightdecay(w,performance,r_pre,r_post,rate,...
%                                             alpha);

%         delta_w = hebbian_rule(W_train,r_pre,r_post,rate);
%     
%     delta_w = oja_rule(W_train, mean(r(:,interval-200:interval),2), ...
%         mean(x(:,interval-200:interval),2), 0.01);
    
    W_train = W_train+delta_w;
    % Normalize so every unit in the RNN gets inputs of norm = 1
%     for j = 1:size(W_train,1)
%          W_train(j,:) = W_train(j,:)/norm(W_train(j,:));
%     end
    
    SSE(trial) = sum(sum((Y-(R'*W_train')).^2,1));
end
%% Plot Performance over training

figure
ax = gca;
plot([1:trials],SSE)
xlabel('Iteration')
ylabel('Performance (SSE)')
makeNiceFigure(ax)
title('training performance')

figure
ax=gca;
plot(performance)
makeNiceFigure(ax)
title('performance')

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
        plot(y,'b','LineWidth',2)
    elseif context == 1 && correct_ans(trial) == -1
        plot(y,'m','LineWidth',2)
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
