clear all
close all

%% Make Integrating network
single = true;

% Number of Neurons In each population
dims = 10;

if single
    % Single Integrating Network Case
    
    % Integrating Population, recurrent weights
    W_rec = abs(randn(dims,dims)*(1/sqrt(dims))); % Added 1/srt(dims)
    for i = 1:dims
        W_rec(:,i) = W_rec(i,:)/sum(W_rec(i,:));
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
dims_input = 10;

% Initialize weights for stimuli and context (just using stimuli for now)
alpha_c = randn(dims_input,1)*1/sqrt(dims_input);
alpha_m = randn(dims_input,1)*1/sqrt(dims_input);
alpha_ctx1 = rand(dims_input,1)-0.5; % only allow 50% modulation.
alpha_ctx2 = rand(dims_input,1)-0.5;
% a_0 = randn(dims_input,1)+4;
a_0 = zeros(dims_input,1);

lambda = 0.001;
lambda = sqrt(lambda);

% Define conditions
context = [1,2];
ctx1 = [1,0];
ctx2 = [0,1];

coherence_values = [0.1 0.3 -0.1 -0.3];
motion_coherence_values = coherence_values;
color_coherence_values = coherence_values;

% Initialize design matrix
numconds = length(ctx1)*length(coherence_values)*length(coherence_values); % all permutations 
R = zeros(dims_input,numconds);
Y = zeros(numconds,dims);
lambda_mat = eye(numconds,numconds)*lambda;

% Build design matrix
for dim = 1:dims
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
                Y(n,dim) = ((color*ctx1(ctx))+(motion*ctx2(ctx)));
                
                % Multiply by component of integrating eigenvector 
                Y(n,dim) = Y(n,dim)*eig_1(dim);
                n = n+1;
            end
        end
    end
end

% R = [R;lambda_mat];
% Y = [Y;zeros(numconds,dims)];

%% Regress to get the optimal output weights
for output = 1:dims
    W(:,output) = regress(Y(:,output),R(:,:)');
end

ridge_param = 1e-4;
for output = 1:dims
    W_ridge(:,output) = ridge(Y(:,output),R(:,:)',ridge_param);
end

%% Check Performance
SSE = sum(sum((Y-(R'*W_ridge)).^2,1)); %Checked by hand. R'*W = W'*R
if SSE<eps
    disp('SSE tiny, regression converged')
else
    SSE
end

%% Define Temporal parameters

% Integration time (ms)
interval = 1300;

% Biological time constant
t_bio = 10; %rate changes are scaled to 10% of what they would be 

integration_dt = t_bio / 100; %integration timestep. just slows things down

% Timesteps for plotting
timesteps = 1:interval;

%% Run simulations with aligned input weights
figure
trials = 1;

% context = [1,2];

context = 1;
ctx1 = [1,0];
ctx2 = [0,1];

coherence_values = [0.05 -0.05];
motion_coherence_values = coherence_values;
color_coherence_values = coherence_values;

noise_std = .4;

% noise_norm = sqrt(2*t_bio*(1/sqrt(dims)));
noise_norm = 0;

% Initialize rates for network
x = zeros(dims,1);

for n = 1:trials
    
    for i = 1:interval
        
        % Stimulus Period is 750ms in the middle of trial
        if i>250 && i<1000
            % Stimuli + noise
            color = color_coherence_values(1) + randn(1)*noise_std;
            motion = motion_coherence_values(2)+ randn(1)*noise_std;
%             motion = 0;

        else
            color = 0;
            motion = 0;
        end
        
        % Rates of input layer
        r(:,i) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
            alpha_ctx2,context,a_0);
        
        % Change of rates in integrator network
        dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + W_ridge'*r(:,i)+noise_norm*randn(size(r,1),1))/t_bio;
        
        % Update rates
        x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
        
        % Readout of integrator
        y(i) = eig_1'*x(:,i);
    end
    
    plot(y,'k','LineWidth',2)
    hold on
    xlabel('timesteps')
    ylabel('Integrated value')
    axis([0 interval -1 1])
    
%     % Estimate performance
%     if max(y) >=0.5
%         choice(n) = 1; %Choose right
%     elseif min(y) <= -0.5
%         choice(n) = 0; %Choose left (didn't choose right)
%     end
    
    % Simpler Estimate performance
    if sign(y(interval)) >=0
        choice(n) = 1; %Choose right
    elseif sign(y(interval)) < 0
        choice(n) = 0; %Choose left (didn't choose right)
    end
    
    
    %     random_proj(n) = eig_1'*M_weights;
end

% timesteps_all = repmat(timesteps,dims_input,1);
%% Firing Rates of Input Neurons
figure
for j = 1:size(r,1)
    plot(timesteps,r(j,:))
    hold on
end
axis([0 interval -1 10])
%% Firing Rates of Output Neurons

figure
for j = 1:size(x,1)
    plot(timesteps,x(j,1:end-1))
    hold on
end


%% Tuning curves for input neurons

% Define a range to test input over.
stimulus_range = -10:.1:10;

% Get range for motion
for i = 1:length(stimulus_range)
    
    color = 0;
    motion = stimulus_range(i);
    context = 0;
    
    tuning_curves_motion(:,i,1) = input_rate(alpha_c,color,alpha_m,...
        motion,alpha_ctx1,alpha_ctx2,context,a_0);
    
end

% Plot single feature tuning curves
figure 
for j = 1:dims_input
plot(stimulus_range,tuning_curves_motion(j,:))
hold on
end
axis([min(stimulus_range) max(stimulus_range) -0.5 max(max(tuning_curves_motion))])


% Get range for color
for i = 1:length(stimulus_range)
    
    color = stimulus_range(i);
    motion = 0;
    context = 0;
    
    tuning_curves_color(:,i) = input_rate(alpha_c,color,alpha_m,motion,...
        alpha_ctx1,alpha_ctx2,context,a_0);
    
end

% Plot single feature tuning curves

figure 
for j = 1:dims_input
plot(stimulus_range,tuning_curves_color(j,:))
hold on
end
axis([min(stimulus_range) max(stimulus_range) -0.5 max(max(tuning_curves_color))])

% Get the joint tuning of color and motion
for i = 1:length(stimulus_range)
    for k = 1:length(stimulus_range)
    
    color = stimulus_range(i);
    motion = stimulus_range(k);
    context = 0;
    
    joint_tuning(i,k,:) = input_rate(alpha_c,color,alpha_m,motion,...
        alpha_ctx1,alpha_ctx2,context,a_0);
    end
end

% Plot a representative sample of joint tuning curves
figure
subplot(2,2,1)
imagesc(joint_tuning(:,:,1))
xlabel('Motion Strength')
ylabel('Color Strength')
subplot(2,2,2)
imagesc(joint_tuning(:,:,2))
xlabel('Motion Strength')
ylabel('Color Strength')
subplot(2,2,3)
imagesc(joint_tuning(:,:,3))
xlabel('Motion Strength')
ylabel('Color Strength')
subplot(2,2,4)
imagesc(joint_tuning(:,:,4))
xlabel('Motion Strength')
ylabel('Color Strength')

 %% Learning Rule
% 
% % Implement hebbian, reward based learning rule. Try to stop before fully
% % trained to demonstrate advantage of higher dimensions. 
% 
% % Number of trials to train on. 
% trials = 100;
% 
% % Number of Neurons In each population
% dims = 100;
% 
% % Randomly initialize weights
% W_init = randn(dims,dims_input);
% 
% % Ensure that all ouput neurons recieve input on a weight vector with norm
% % = 1. Make sure all neurons are generally getting the same amount of
% % input. 
% for j = 1:size(W_init,1)
%     W_init(j,:) = (W_init(j,:))/norm(W_init(j,:));
% end
% 
% % Make a look-up table for trial order, randomized to improve learning both
% % contexts simultaneously. 
% context = [1 2];
% coherence_values = [0.1 -0.1];
% motion_coherence_values = coherence_values;
% color_coherence_values = coherence_values;
% 
% for trial = 1:trials
% idx_m = randperm(length(motion_coherence_values));
% idx_c = randperm(length(color_coherence_values));
% idx_ctx = randperm(length(context));
% 
% trial_conds(trial,:) = [idx_ctx(1) idx_c(1) idx_m(1)];
% 
% stim_sign = [sign(color_coherence_values(idx_c(1)))...
%     sign(motion_coherence_values(idx_m(1)))];
% 
% correct_ans(trial) = stim_sign(idx_ctx(1));
% end
% 
% % Define noise to apply to the sensory signals
% noise_std = 0;
% 
% % Initialize rates for network
% x = zeros(dims,1);
% W_train = W_init;
% 
% for trial = 1:trials
%     
%     for i = 1:interval
%         
%         context = trial_conds(trial,1);
%         % Stimulus Period is 500ms in the middle of trial
%         if i>250 
%             % Stimuli + noise
%             color = color_coherence_values(trial_conds(trial,2)) + randn(1)*noise_std;
%             motion = motion_coherence_values(trial_conds(trial,3))+ randn(1)*noise_std;
% %             motion = 0;
% 
%         else
%             color = 0;
%             motion = 0;
%         end
%         
%         % Rates of input layer
%         r(:,i) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
%             alpha_ctx2,context,a_0);
%         
%         % Change of rates in integrator network
%         dx_dt(:,i)= (-x(:,i) + W_rec*x(:,i) + W_train'*r(:,i))/t_bio;
%         
%         % Update rates
%         x(:,i+1) = x(:,i)+dx_dt(:,i)*integration_dt;
%         
%         % Readout of integrator
%         y(i) = eig_1'*x(:,i);
%     end
%     
%     plot(y,'k','LineWidth',2)
%     hold on
%     xlabel('timesteps')
%     ylabel('Integrated value')
%     
%     % Simpler Estimate of performance
%     if sign(mean(y(interval-200:interval))) >=0
%         choice(trial) = 1; %Choose right
%     elseif sign(y(interval)) < 0
%         choice(trial) = -1; %Choose left (didn't choose right)
%     end
%     
%     % Compute trial performance
%     if choice(trial) == correct_ans(trial)
%         performance(trial) = 1*mean(y(interval-200:interval));
%     else
%         performance(trial) = -1*mean(y(interval-200:interval));
%     end
%     
%     %Update Weights
%     delta_w = learning_rule(performance, mean(r(:,interval-200:interval),2), ...
%         mean(x(:,interval-200:interval),2), 0.5);
%     
%    
%    
% end
% 
% 
