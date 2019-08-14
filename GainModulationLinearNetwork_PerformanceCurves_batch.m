clear all
close all

%% Make Integrating network
single = true;
runs = 20;
dimset = [10 50 100 500];
for dimcond = 1:length(dimset)
    proportion_choose_green_runs = [];
    proportion_choose_right_runs = [];
    overall_performance = [];
    for run = 1:runs
        % Number of Neurons In each population
        % dims = 800;
        dims = dimset(dimcond);
        
        if single
            % Single Integrating Network Case
            
            % Integrating Population, recurrent weights
            W_rec = abs(randn(dims,dims)*(1/sqrt(dims))); % Added 1/srt(dims)
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
%         if ~single
%             plot_eig(D_L)
%             plot_eig(D_R)
%         else
%             plot_eig(D)
%         end
        % Get the eigenvector along the integrating dimension
        if ~single
            eig_1_L = get_int_eig(V_L,D_L);
            eig_1_R = get_int_eig(V_R,D_R);
        else
            eig_1 = get_int_eig(V,D);
        end
        
        %% Define sensory layer rates
        
        % Size of input layer
%         dims_input = 50;
        dims_input = dimset(dimcond);
        
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
        
        coherence_values = [-0.15 -0.036 -0.009 0.009, 0.036, 0.15];
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
        W = [];
        W_ridge = [];
        
        for output = 1:dims
            W(:,output) = regress(Y(:,output),R(:,:)');
        end
        
        % ridge_param = 5e-3;
        ridge_param = 5e-6;
        for output = 1:dims
            W_ridge(:,output) = ridge(Y(:,output),R(:,:)',ridge_param);
        end
        
        % Flip them around so they are useful
        W = W';
        W_ridge = W_ridge';
        % W_ridge(:,1) = [];
        
%         figure
%         subplot(1,2,1)
%         imagesc(W)
%         colorbar
%         subplot(1,2,2)
%         imagesc(W_ridge)
%         colorbar
        
        
        %% Normalize input weights
        
%         for idx = 1:size(W,1)
%             W_norm(idx,:) = W(idx,:)/norm(W(idx,:));
%         end
%         
%         
%         
%         for idx = 1:size(W_ridge,1)
%             W_ridge_norm(idx,:) = W_ridge(idx,:)/norm(W_ridge(idx,:));
%         end
        
%         figure
%         subplot(1,2,1)
%         imagesc(W_norm)
%         colorbar
%         subplot(1,2,2)
%         imagesc(W_ridge_norm)
%         colorbar
        %
        %% Check Performance
        % SSE = sum(sum((Y-(R'*W_ridge')).^2,1)); %Checked by hand. R'*W = W'*R
        % if SSE<eps
        %     disp('SSE tiny, regression converged')
        % else
        %     SSE
        % end
        
        %% Set up testing batch
        
        % Make a look-up table for trial order, randomized to improve learning both
        % contexts simultaneously.
        context = [1 2];
        coherence_values = [-0.15 -0.036 -0.009 0.009, 0.036, 0.15];
        motion_coherence_values = coherence_values;
        color_coherence_values = coherence_values;
        conditions = length(context)*length(coherence_values)^2;
        repeats = 4;
        trials = conditions*repeats;
        trial_conds=zeros(conditions,4);
        correct_ans = [];
        
        n = 1;
        for i = 1:length(context)
            for j = 1:length(motion_coherence_values)
                for k = 1:length(color_coherence_values)
                    idx_m = motion_coherence_values(k);
                    idx_c = color_coherence_values(j);
                    idx_ctx = context(i);
                    
                    % Define the conditions for each trial, [context, color, motion]
                    trial_conds(n,1:3) = [idx_ctx idx_c idx_m];
                    
                    stim_sign = [sign(idx_c) sign(idx_m)];
                    
                    % Define the correct answer for each trial
                    correct_ans(n) = stim_sign(idx_ctx);
                    trial_conds(n,4) = correct_ans(n);
                    n=n+1;
                end
            end
        end
        
        trial_conds = repmat(trial_conds,repeats,1);
        
        %% Define Temporal parameters
        
        % Integration time (ms)
        interval = 1300;
        
        % Biological time constant
        t_bio = 10; %rate changes are scaled to 10% of what they would be
        
        integration_dt = t_bio / 100; %integration timestep. just slows things down
        
        % Timesteps for plotting
        timesteps = 1:interval;
        
        %% Run simulations with aligned input weights
        
        % Define noise to apply to the sensory signals
        noise_std = 2;
        
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
        correct_ans = zeros(trials);
        
        % Noise for RNN
        rho = 0.1;
        
        for trial = 1:trials
            % Update of progress
            clc
            disp(['iteration number ' num2str(trial)])
            disp(['proportion correct = ' num2str(propCorrect)])
            
            % Initialize dynamic parameters before each trial
            r = zeros(dims_input,interval);
            dx_dt = zeros(dims,interval);
            x = zeros(dims,interval);
            
            % Iterate through condition combinations
            ctx = trial_conds(trial,1);
            
            % Define the correct answer for each trial
            correct_ans(trial) = trial_conds(trial,4);
            
            % Run the trial
            for t = 1:interval
                
                % Stimulus Period is 750ms in the middle of trial
                if t>250 && t<1000
                    % Stimuli + noise
                    color = trial_conds(trial,2) + randn(1)*noise_std;
                    motion = trial_conds(trial,3)+ randn(1)*noise_std;
                    %             motion = 0;
                    
                else
                    color = 0;
                    motion = 0;
                end
                
                % Rates of input layer
                r(:,t) = input_rate(alpha_c,color,alpha_m,motion,alpha_ctx1,...
                    alpha_ctx2,ctx,a_0);
                
                % Change of rates in integrator network
                dx_dt(:,t)= (-x(:,t) + W_rec*x(:,t) + W*r(:,t)+...
                    rho*randn(size(x,1),1))/t_bio;
                
                % Update rates
                x(:,t+1) = x(:,t)+dx_dt(:,t)*integration_dt;
                
                % Readout of integrator
                y(trial,t) = eig_1'*x(:,t);
            end
            
            choice(trial) = sign(y(trial,end));
            performance_vec(trial) = choice(trial)*correct_ans(trial);
            
            propCorrect = numel(find(performance_vec==1))/length(performance_vec);
        end
        performance_vec = performance_vec';
        simulation_data = [trial_conds performance_vec];
        
        %% Work out the proportion correct for each condition
        
        % Deal with the -1 error sign.
        performance_vec(find(performance_vec==-1)) = 0;
        
        % color_mask=[];
        color_performance=[];
        % motion_mask = [];
        motion_performance =[];
        mean_color_performance = [];
        mean_motion_performance = [];
        
        % Marginalize accross motion to get color
        for i = 1:length(context)
            for k = 1:length(color_coherence_values)
                color_mask(:,k,i) = trial_conds(:,1)==i & trial_conds(:,2)==color_coherence_values(k);
                color_performance(:,k,i) = performance_vec(color_mask(:,k,i));
                color_choice(:,k,i) = choice(color_mask(:,k,i));
            end
        end
        
        % Marginalize accross color to get motion
        for i = 1:length(context)
            for j = 1:length(motion_coherence_values)
                motion_mask(:,j,i) = trial_conds(:,1)==i & trial_conds(:,3)==motion_coherence_values(j);
                motion_performance(:,j,i) = performance_vec(motion_mask(:,j,i));
                motion_choice(:,j,i) = choice(motion_mask(:,j,i));
            end
        end
        
        % Display the performance values for this run.
        coherence_values
        mean_color_performance = mean(color_performance,1)
        mean_motion_performance = mean(motion_performance,1)
        proportion_choose_green = sum(color_choice==1,1)/size(color_choice,1)
        proportion_choose_right = sum(motion_choice==1,1)/size(motion_choice,1)
        
        %% Runs Across different initializations
        
        
        if run==1
            proportion_choose_green_runs = proportion_choose_green;
            proportion_choose_right_runs = proportion_choose_right;
        else
            proportion_choose_green_runs = (proportion_choose_green_runs+proportion_choose_green)./2 ;
            proportion_choose_right_runs = (proportion_choose_right_runs+proportion_choose_right)./2 ;
        end
        
        if run==1
            overall_performance = propCorrect;
        else
            overall_performance = (overall_performance+propCorrect)/2;
        end
        
        % proportion_green = mean_color_performance;
        % proportion_right = mean_motion_performance;
        %
        % proportion_green(:,coherence_values<0,:) = 1-mean_color_performance(:,coherence_values<0,:);
        % proportion_right(:,coherence_values<0,:) = 1-mean_motion_performance(:,coherence_values<0,:);
    end % end Runs
    
    
    proportion_choose_green_dim(:,:,:,dimcond) = proportion_choose_green_runs;
    proportion_choose_right_dim(:,:,:,dimcond) = proportion_choose_right_runs;
    overall_performance_dim(:,dimcond) = overall_performance;
    
    
end % end Dims


%% Plot performance as a function of dim

dimset = [10 50 100 500];

% figure
% ax = gca;
for dimcond = 1:length(dimset)
[param(dimcond,:),~,f] = sigm_fit(coherence_values,proportion_choose_green_dim(:,:,1,dimcond));
x_vector=min(coherence_values):(max(coherence_values)-...
    min(coherence_values))/100:max(coherence_values);
end
% plot(x_vector,f(param,x_vector),'LineWidth',3)
% hold on
% dimstr(dimcond,:) = ['dimensions = ' num2str(dimset(1))];

% legend(num2str(dimset(1)),num2str(dimset(2)),num2str(dimset(3)))
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)

figure
ax = gca;
for dimcond = 1:length(dimset)
plot(x_vector,f(param(dimcond,:),x_vector),'LineWidth',3)
% scatter(coherence_values,proportion_choose_green_dim(:,:,1,dimcond))
hold on
end
legend(num2str(dimset(1)),num2str(dimset(2)),num2str(dimset(3)))
axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
makeNiceFigure(ax)
% plot(coherence_values,proportion_choose_green_dim(:,:,1,:),'k.','MarkerSize',30)
% 
% figure
% ax = gca;
% for dimcond = 1:length(dimset)
% X = [ones(length(coherence_values),1) coherence_values'];
% b = X\proportion_choose_green_dim(:,:,2,dimcond)';
% yCalc2 = X*b;
% plot(coherence_values,yCalc2,'LineWidth',3)
% hold on
% end
% legend(num2str(dimset(1)),num2str(dimset(2)),num2str(dimset(3)),num2str(dimset(4)))
% % plot(coherence_values,proportion_choose_green_runs(:,:,2),'k.','MarkerSize',30)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)
% 
% figure
% ax = gca;
% for dimcond = 1:length(dimset)
% X = [ones(length(coherence_values),1) coherence_values'];
% b = X\proportion_choose_right_dim(:,:,1,dimcond)';
% yCalc2 = X*b;
% plot(coherence_values,yCalc2,'LineWidth',3)
% hold on
% end
% % plot(coherence_values,proportion_choose_right_runs(:,:,1),'k.','MarkerSize',30)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% legend(num2str(dimset(1)),num2str(dimset(2)),num2str(dimset(3)),num2str(dimset(4)))
% makeNiceFigure(ax)
% 
% figure
% ax = gca;
% for dimcond = 1:length(dimset)
% [param,~,f] = sigm_fit(coherence_values,proportion_choose_right_dim(:,:,2,dimcond));
% x_vector=min(coherence_values):(max(coherence_values)-...
%     min(coherence_values))/100:max(coherence_values);
% plot(x_vector,f(param,x_vector),'LineWidth',3)
% hold on
% end
% legend(num2str(dimset(1)),num2str(dimset(2)),num2str(dimset(3)),num2str(dimset(4)))
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)

 %% Plot performance as a function of context
% figure
% ax = gca;
% [param,~,f] = sigm_fit(coherence_values,proportion_choose_green_runs(:,:,1));
% x_vector=min(coherence_values):(max(coherence_values)-...
%     min(coherence_values))/100:max(coherence_values);
% plot(coherence_values,proportion_choose_green_runs(:,:,1),'k.', ...
%     x_vector,f(param,x_vector),'b-','MarkerSize',30,'LineWidth',3)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)
% 
% figure
% ax = gca;
% X = [ones(length(coherence_values),1) coherence_values'];
% b = X\proportion_choose_green_runs(:,:,2)';
% yCalc2 = X*b;
% plot(coherence_values,yCalc2,'b-','LineWidth',3)
% hold on
% plot(coherence_values,proportion_choose_green_runs(:,:,2),'k.','MarkerSize',30)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)
% 
% figure
% ax = gca;
% X = [ones(length(coherence_values),1) coherence_values'];
% b = X\proportion_choose_right_runs(:,:,1)';
% yCalc2 = X*b;
% plot(coherence_values,yCalc2,'b-','LineWidth',3)
% hold on
% plot(coherence_values,proportion_choose_right_runs(:,:,1),'k.','MarkerSize',30)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)
% 
% figure
% ax = gca;
% [param,~,f] = sigm_fit(coherence_values,proportion_choose_right_runs(:,:,2));
% x_vector=min(coherence_values):(max(coherence_values)-...
%     min(coherence_values))/100:max(coherence_values);
% plot(coherence_values,proportion_choose_right_runs(:,:,2),'k.', ...
%     x_vector,f(param,x_vector),'b-','MarkerSize',30,'LineWidth',3)
% axis([coherence_values(1)-.01 coherence_values(end)+.01 -0.1 1.1])
% makeNiceFigure(ax)
