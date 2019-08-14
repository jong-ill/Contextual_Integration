% Learning Rule

% Implement hebbian, reward based learning rule. Try to stop before fully
% trained to demonstrate advantage of higher dimensions.

clear all
close all

%% Set up populations and conditions

% Number of trials to train on.
trials = 100;

% Number of neurons in integrating population
dims = 100;

% Integrating Population, recurrent weights
[W_rec,V,D] = make_int_excitatory(dims);

% Define integrating eigenvector
eig_1 = get_int_eig(V,D);

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
    for i = 1:length(context)
        for j = 1:length(motion_coherence_values)
            for k = 1:length(color_coherence_values)
                
                % Iterate through condition combinations
                motion = motion_coherence_values(j);
                color = color_coherence_values(k);
                ctx = context(i);
                
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

%% Setup sensory layer-> integrating population

% Randomly initialize weights of input layer-> integrating population
W_init = randn(dims,dims_input);

% Ensure that all ouput neurons recieve input on a weight vector with norm
% = 1. Make sure all neurons are generally getting the same amount of
% input.
for j = 1:size(W_init,1)
    W_init(j,:) = (W_init(j,:))/norm(W_init(j,:));
end

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
noise_std = 0.2;

% Initialize rates for network
x = zeros(dims,1);

% W_train begins as the initial weights and is trained by learning
W_train = W_init;

%% Define Temporal parameters

% Integration time
interval = 1000;

% Biological time constant
t_bio = 10;
integration_dt = t_bio / 100; %integration timestep

% Timesteps for plotting
timesteps = 1:interval;

%% Run Training

for trial = 1:trials
    
    for i = 1:interval
        
        context = trial_conds(trial,1);
        % Stimulus Period is 500ms in the middle of trial
        if i>250
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
        r(:,i) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
            alpha_ctx2,context,a_0);
        
        % Change of rates in integrator network
        dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + W_train*r(:,i))/t_bio;
        
        % Update rates
        x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
        
        % Readout of integrator
        y(i) = eig_1'*x(:,i);
    end
    
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
    xlabel('timesteps')
    ylabel('Integrated value')
    axis([0 interval -1 1])
%     plot(1:1000,repmat(0,1000))
    
    % Simpler Estimate of performance
    if sign(mean(y(interval-200:interval))) >=0
        choice(trial) = 1; %Choose right
    elseif sign(y(interval)) < 0
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
        performanace(trial) = sign(performance(trial));
    end
    
    eta = 0.15;
    alpha = 0.5;
    
    %Update Weights
    delta_w = hebbianrule_weightdecay(performance(trial), ...
                            mean(r(:,interval-200:interval),2), ...
                            mean(x(:,interval-200:interval),2), ...
                            eta, alpha, W_train);
    
    W_train = W_train+delta_w;
    
end
    plot(1:1000,repmat(0,1000))

%% Check Performance
SSE = sum(sum((Y-(R'*W_train')).^2,1));
if SSE<eps
    disp('SSE tiny, regression converged')
else
    SSE
end
