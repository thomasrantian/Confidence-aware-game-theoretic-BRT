addpath('/Users/rantian/Dropbox/Confidence_human_aware_reachability/roundabout/roundabout_map/');
addpath('/Users/rantian/Dropbox/npy-matlab/npy-matlab')

close all;
clear global;
clear;
global data plots_handle BRT_all_2c grid show_frenet_BRT BRT_all_true sim_log aug_sampled_policy_set Acc_sampled_policy_set env_params safe_yawrate_all_2c safe_acc_all_2c;

if isempty(data)
    data = load('/Users/rantian/Dropbox/Confidence_human_aware_reachability/roundabout/roundabout_data/interact_data_whole_1003.mat');
    data = data.interact_data_new;
end
test_cases = [117,  110, 200, 3, 40, -1, 0, 1.5, 2, 0, 1]; 
%% Config params
env_params.dt            = 0.1;          
env_params.horizon       = 17;
env_params.measur_length = 17;
% traffic eval
env_params.f_weights = [36, 4, 3, 0, 0, 0];
env_params.f_weights = env_params.f_weights / 200;
env_params.l_weights = [36, 4, 3, 0, 0, 0];
env_params.l_weights = env_params.l_weights / 200;
env_params.ref_weight = 0.05;
env_params.r_weights = [0, 20, 3, 0, 0, -1];
env_params.r_weights = env_params.r_weights / 100;
env_params.ref_weight_r = 0.05;
env_params.ah_weights = [-10, 0, 3, 0, -10, 2];
env_params.ah_weights = env_params.r_weights / 100;
env_params.ref_weight_ah = 0.00;
env_params.ih_weights = [-5, 20, 3, 0, 0, -1];
env_params.ih_weights = env_params.r_weights / 100;
env_params.ref_weight_ah = 0.00;
env_params.l_car            = 5;
env_params.w_car            = 2.5;
env_params.decay            = 1 ; % accumulative reward discount factor

env_params.beta_set         = [0.01, 5, 20]; % model confidence set
env_params.acc_max          = 5;
env_params.acc_min          = -2;

env_params.yaw_rate_max     = 0.6;
env_params.yaw_rate_min     = -0.6;

env_params.v_max            = 8;

boundary_value              = 0.6;   
buffer_scale                = 1.0;
ctrl_bound_Delta            = 0.85;

% state grid used for the BRT
grid.v_grid                 = linspace(-1, 10, 22);
grid.phi_grid               = linspace(-180, 180, 24);
grid.acc_grid               = linspace(-2, 2, 9);
grid.yawrate_grid           = linspace(-0.5, 0.5, 4);

plots_handle                = {};  % store all the plot handles
n_fig                       = 1;
run_sim                     = 1; % flag to run the simulation
show_frenet_BRT             = 0; % show the BRT in the frenet frame
verify_state_update         = 0;
show_leader_follower_path   = 0;

% simulated human
human_mode       = "follower" ;
human_beta_index = 3;
log_sim_mode     = 0;
safety_mode      = 2;
Bayesian_BRT_mode = "full";

%% Load BRT
load('BRT_map.mat');

da              = 0.2;
a_0_grid        = -2: da: 2;

Acc_policy_set  = cell(length(a_0_grid),1);
Acc_param_set   = cell(length(a_0_grid),1);


Acc_sampled_policy_set = load('Acc_sampled_policy_set.mat');
Acc_sampled_policy_set = Acc_sampled_policy_set.Acc_sampled_policy_set;


d_yaw_rate  = 0.05;
yaw_rate_grid = -0.2: d_yaw_rate: 0.5;

yaw_rate_policy_set = cell(length(yaw_rate_grid),1);
yaw_rate_param_set  = cell(length(yaw_rate_grid),1);

yaw_rate_sampled_policy_set = load('yaw_rate_sampled_policy_set.mat');
yaw_rate_sampled_policy_set = yaw_rate_sampled_policy_set.yaw_rate_sampled_policy_set;


aug_sampled_policy_set = load('aug_sampled_policy_set.mat');
aug_sampled_policy_set = aug_sampled_policy_set.aug_sampled_policy_set;

%% Main sim loop

if run_sim
    
    for test_id = 1:1:size(test_cases,1)
       
        % plot the map
        figure(n_fig);
        hold on; box on;
        axis([980 1050, 960 1030])
        set(gcf,'units','centimeters','position',[10,10,30,30])
        axis equal;
        [maps.CurbPts, maps.LanePts, maps.StopPts] = osmXYParserFun('DR_USA_Roundabout_FT.osm_xy', 1, 1); 
        % extract an interaction traj from the data set
        traj = extract_traj(test_cases(test_id,1),test_cases(test_id,2),test_cases(test_id,3));
        human_car_index = test_cases(test_id,4) - 1;
        robot_car_index = mod(human_car_index,2) + 1;
        % saving the observed human acctions
        human_observation_buffer = [];
        sim_log = cell(test_cases(test_id,5), 1);
        b_latent = ones(1, 2 * length(env_params.beta_set)) / 2/length( env_params.beta_set);
        b_latent_hist = [b_latent];
        % initialize the world state
        s_w = zeros(7,2); % [x,y,theta,kappa,v,a,yaw_rate]
        s_w(:,human_car_index) = getWorldState(human_car_index, test_cases(test_id, 11), traj);
        s_w(:,robot_car_index) = getWorldState(robot_car_index, 1, traj);
        if ~log_sim_mode
            s_w(6,human_car_index) =  test_cases(test_id,6);
            s_w(6,robot_car_index) =  test_cases(test_id,7);
            if ~isnan(test_cases(test_id,8))
                s_w(5,human_car_index) =  test_cases(test_id,8);
            end
            if ~isnan(test_cases(test_id,9))
                s_w(5,robot_car_index) =  test_cases(test_id,9);
            end
        end
        
        [rob_ref, human_ref] = getRefLane(traj, human_car_index);
        rob_ref_length = getPathLength(rob_ref);
        human_ref_length = getPathLength(human_ref);
        full_brt_violation = 0;
        our_brt_violation = 0;
 
        for t = 1:1:test_cases(test_id,5)
            % visualize the cars
            if show_frenet_BRT
                subplot(n_fig, 2, 1);
            end
            plots_handle{end + 1} = plot_car([s_w(1,1),s_w(2,1),s_w(3,1)/pi*180], "car-orange.png", 1, env_params.l_car, env_params.w_car);
            plots_handle{end + 1} = plot_car([s_w(1,2),s_w(2,2),s_w(3,2)/pi*180], "car-red.png",    1, env_params.l_car, env_params.w_car);
            sim_log{t,1}.s_w = s_w;
            % compute the relative state
            rel5didx = getRelativeStateIndex(s_w, human_car_index);
            % plot the full BRT
%             if isnan(safe_acc_full)
%                 issafe_full = 1;
%             end
%             if ~issafe_full
%                 full_brt_violation = full_brt_violation + 1;
%             end

            % extract the motion plan of the robot
            acc_robot_plan = zeros(1, env_params.horizon);
            for k = 0:1:env_params.horizon-1
                acc_robot_plan(1,k+1) = traj{robot_car_index,1}(5,t+k);
            end
            yaw_rate_robot_plan = zeros(1, env_params.horizon);
            for k = 0:1:env_params.horizon-1
                yaw_rate_robot_plan(1,k+1) = traj{robot_car_index,1}(6,t+k);
            end
            if verify_state_update
                if show_frenet_BRT
                    subplot(n_fig, 2, 1);
                end
                sw_rob_future = [];
                sw_robot_temp = s_w(:,robot_car_index)';
                for k = 1:1:env_params.horizon
                    sw_robot_temp = updateState(sw_robot_temp, [acc_robot_plan(k), yaw_rate_robot_plan(k)], env_params);
                    sw_rob_future = [sw_rob_future; sw_robot_temp([1,2,3,5])];
                end
                plots_handle{end + 1} = plot(sw_rob_future(:,1), sw_rob_future(:,2),'ro','MarkerSize', 10);
                plots_handle{end + 1} = plot(traj{robot_car_index,1}(1,t+1:t+env_params.horizon), traj{robot_car_index,1}(2,t+1:t+env_params.horizon),'kd','MarkerSize', 10);
            end
            
            acc0_class = min(max(round((s_w(6,human_car_index) - a_0_grid(1)) / da + 1),1), length(a_0_grid));
            acc0_class = 11;
            yawrate0_class = min(max(round((s_w(7,human_car_index) - yaw_rate_grid(1)) / d_yaw_rate + 1),1), length(yaw_rate_grid));
            yawrate0_class = 6;
            acc0_class_rob = min(max(round((s_w(6,robot_car_index) - a_0_grid(1)) / da + 1),1), length(a_0_grid));
            yawrate0_class_rob = min(max(round((s_w(7,robot_car_index) - yaw_rate_grid(1)) / d_yaw_rate + 1),1), length(yaw_rate_grid));
            yawrate0_class_rob = 6;
            policy_set_human = aug_sampled_policy_set{acc0_class, yawrate0_class};
            policy_set_rob  = aug_sampled_policy_set{acc0_class_rob, yawrate0_class_rob};
            
            Boltzmann_plan_prob_set = getBoltzmannPolicy(s_w, robot_car_index, nan,...
                                                        rob_ref, env_params, policy_set_rob,traj, t,'r');
            maximum = max(max(Boltzmann_plan_prob_set));
            [best_boltzmann_beta_idx, best_boltzmann_plan_idx]=find(Boltzmann_plan_prob_set==maximum);
            best_boltzmann_plan_idx = best_boltzmann_plan_idx(1);
          
            robot_motion = buildMotion(s_w,robot_car_index, env_params, policy_set_rob{best_boltzmann_plan_idx(1)});
            
            plots_handle{end + 1} = plot(robot_motion(:,1), robot_motion(:,2),'c*','MarkerSize', 5,'DisplayName','Robot motion');            
            
            
            Boltzmann_plan_prob_set_h = getBoltzmannPolicy(s_w, human_car_index, nan,...
                                                        human_ref, env_params, policy_set_human,traj, t,'ah');
            maximum = max(max(Boltzmann_plan_prob_set_h));
            [best_boltzmann_beta_idx_h, best_boltzmann_plan_idx_h]=find(Boltzmann_plan_prob_set_h==maximum);
            best_boltzmann_plan_idx_h = best_boltzmann_plan_idx_h(1);
          
            human_motion = buildMotion(s_w,human_car_index, env_params, policy_set_human{best_boltzmann_plan_idx_h(1)} * 0 + [1;0]);
 
            isolated_human_plan_prob_set_h = getBoltzmannPolicy(s_w, human_car_index, nan,...
                                                        human_ref, env_params, policy_set_human,traj, t,'r');
            maximum = max(max(isolated_human_plan_prob_set_h));
            [best_boltzmann_beta_idx_ih, best_boltzmann_plan_idx_ih]=find(isolated_human_plan_prob_set_h==maximum);
            best_boltzmann_plan_idx_ih = best_boltzmann_plan_idx_ih(1);
          
            human_motion = buildMotion(s_w,human_car_index, env_params, policy_set_human{best_boltzmann_plan_idx_ih(1)});

            follower_plan_prob_set = getFollowerPolicy(s_w, human_car_index, policy_set_rob{best_boltzmann_plan_idx},...
                                                      human_ref, env_params, policy_set_human,traj, t);

            % get the follower distribution
            maximum = max(max(follower_plan_prob_set));
            [best_follower_beta_idx, best_follower_plan_idx]=find(follower_plan_prob_set==maximum);
            best_follower_plan_idx = best_follower_plan_idx(1);
                      
            leader_plan_prob_set = getLeaderPolicy(s_w, human_car_index, human_ref, yaw_rate_robot_plan, env_params, ...
                policy_set_human, [acc0_class_rob, yawrate0_class_rob ], traj, t);
            % get the leader distribution
            maximum = max(max(leader_plan_prob_set));
            [best_leader_beta_idx,best_leader_plan_idx]=find(leader_plan_prob_set==maximum);
            best_leader_plan_idx = best_leader_plan_idx(1);

            if log_sim_mode
                human_acc_predict_true = traj{human_car_index,1}(5, t:t+env_params.horizon-1);
                human_acc_min = min(human_acc_predict_true);
                human_acc_max = max(human_acc_predict_true);
                
                human_yawrate_predict_true = traj{human_car_index,1}(6, t:t+env_params.horizon-1);
                human_yawrate_min = min(human_acc_predict_true);
                human_yawrate_max = max(human_acc_predict_true);
                [human_yawrate_min_index, human_yawrate_max_index,human_acc_min_index, human_acc_max_index] ...
                    = get_action_bound_idx(human_acc_predict_true, human_yawrate_predict_true);
            else
                if human_mode == "follower"
                    sampled_plan_idx = best_follower_plan_idx;
                    sampled_plan_idx = randsample( [1:1:size(follower_plan_prob_set,2)], 1, true, follower_plan_prob_set(human_beta_index,:) );
                    if human_beta_index == 3
                        sampled_plan_idx = best_follower_plan_idx;
                    end
                elseif human_mode == "leader"
                    sampled_plan_idx = randsample( [1:1:size(leader_plan_prob_set,2)], 1, true, leader_plan_prob_set(human_beta_index,:) );
                elseif human_mode == "aggressive"
                    sampled_plan_idx = randsample( [1:1:size(Boltzmann_plan_prob_set_h,2)], 1, true, Boltzmann_plan_prob_set_h(human_beta_index,:) );
                elseif human_mode == "isolation"
                    sampled_plan_idx = randsample( [1:1:size(isolated_human_plan_prob_set_h,2)], 1, true, isolated_human_plan_prob_set_h(human_beta_index,:) );
                end
         
                human_acc_min = min(policy_set_human{sampled_plan_idx}(1,:));
                human_acc_max = max(policy_set_human{sampled_plan_idx}(1,:));
                
                human_yawrate_min = min(policy_set_human{sampled_plan_idx}(2,:));
                human_yawrate_max = max(policy_set_human{sampled_plan_idx}(2,:));
                
                [human_yawrate_min_index, human_yawrate_max_index, human_acc_min_index, human_acc_max_index] ...
                    = get_action_bound_idx(policy_set_human{sampled_plan_idx}(1,:), policy_set_human{sampled_plan_idx}(2,:));
            end
            
            true_BRT_id = BRT_map(strcat(num2str(human_yawrate_min_index), num2str(human_yawrate_max_index), num2str(human_acc_min_index), num2str(human_acc_max_index)));
            
            if human_mode == "aggressive"
                true_BRT_id = BRT_map(strcat(num2str(2), num2str(3), num2str(1), num2str(9)));
            end
            if log_sim_mode                
                error_set = nan(1, size(policy_set_human, 1));
                parfor plan_id = 1:size(policy_set_human, 1)
                    temp_plan =  policy_set_human{plan_id, 1};
                    plan_motion = buildMotion(s_w,human_car_index, env_params, temp_plan);
                    error = plan_motion(:,1:2)' - traj{human_car_index,1}(1:2,t:t+env_params.horizon- 1 - (env_params.horizon - env_params.measur_length));
                    error = sum(sum(abs(error)));
                    error_set(plan_id) = error;
                end
                [p_, best_match] = min(error_set);
                sw_human_obs_match = [];
                sw_human_temp = s_w(:,human_car_index)';
                for k = 1:1:env_params.horizon
                    sw_human_temp = updateState(sw_human_temp, policy_set_human{best_match, 1}(:,k), env_params);
                    sw_human_obs_match = [sw_human_obs_match; sw_human_temp([1,2,3,5])];
                end
                if show_leader_follower_path
                    plots_handle{end + 1} = plot(sw_human_obs_match(:,1), sw_human_obs_match(:,2),'bd','MarkerSize', 5,'DisplayName','Matched observation');
                end
            else                 
                best_match = sampled_plan_idx;
            end            
            b_latent_pre = b_latent;
            if Bayesian_BRT_mode == "full"
                for role_idx = 1:1:2
                    for beta_idx = 1:1:length(env_params.beta_set)
                        if role_idx == 1
                            P_obs = follower_plan_prob_set(beta_idx, best_match);
                            b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx) = b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx) * P_obs;
                        else
                            P_obs = leader_plan_prob_set(beta_idx, best_match);
                            b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx) = b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx) * leader_plan_prob_set(beta_idx, best_match);
                        end
                    end
                end  
            elseif Bayesian_BRT_mode == "confidence" % only update conidence
                for beta_idx = 1:1:length(env_params.beta_set)
                  
                        P_obs = isolated_human_plan_prob_set_h(beta_idx, best_match);

                        b_latent(beta_idx) = b_latent(beta_idx) * P_obs;
                        
                        b_latent(length(env_params.beta_set) + beta_idx) = b_latent(length(env_params.beta_set) + beta_idx) * P_obs;
                end
            
            elseif Bayesian_BRT_mode == "role" % only update the role
                 for role_idx = 1:1:2
                        if role_idx == 1
                            P_obs = follower_plan_prob_set(3, best_match);
                            b_latent(1) = b_latent(1) * P_obs;
                            b_latent(2) = b_latent(2) * P_obs;
                            b_latent(3) = b_latent(3) * P_obs;
                        else
                            P_obs = leader_plan_prob_set(3, best_match);
                            b_latent(4) = b_latent(4) * P_obs;
                            b_latent(5) = b_latent(5) * P_obs;
                            b_latent(6) = b_latent(6) * P_obs;
                        end
                 end
            end
            b_latent = b_latent / sum(b_latent);
            b_latent = 0.5 * b_latent + 0.5 * b_latent_pre;
            b_latent_hist = [b_latent_hist; b_latent];
            sim_log{t,1}.b_latent = b_latent;
            
            weighted_human_plan_prob_set = zeros(1, length(follower_plan_prob_set));
            if Bayesian_BRT_mode == "full"
                for i = 1:1:length(follower_plan_prob_set)
                    temp = 0;
                    for role_idx = 1:1:2
                        for beta_idx = 1:1:length(env_params.beta_set)
                            if role_idx == 1
                                temp = temp + follower_plan_prob_set(beta_idx, i) *  b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx);
                            else
                                temp = temp + leader_plan_prob_set(beta_idx, i)   *  b_latent((role_idx-1) * length(env_params.beta_set) + beta_idx);
                            end
                        end
                    end
                    weighted_human_plan_prob_set(i) = temp;
                end     
            elseif Bayesian_BRT_mode == "confidence"
                for i = 1:1:length(follower_plan_prob_set)
                    temp = 0;
                    for beta_idx = 1:1:length(env_params.beta_set)
                        temp = temp + isolated_human_plan_prob_set_h(beta_idx, i) *  ( b_latent(beta_idx) + b_latent(length(env_params.beta_set) + beta_idx) );
                    end
                    weighted_human_plan_prob_set(i) = temp;
                end
            elseif Bayesian_BRT_mode == "role" 
                for i = 1:1:length(follower_plan_prob_set)
                    temp = 0;
                    for role_idx = 1:1:2
                        if role_idx == 1
                            temp = temp + follower_plan_prob_set(3, i) *  (b_latent(1) + b_latent(2) + b_latent(3)) ;
                        else
                            temp = temp + leader_plan_prob_set(3, i)   *  (b_latent(4) + b_latent(5) + b_latent(6));
                        end
                    end
                    weighted_human_plan_prob_set(i) = temp;
                end
                                
            end
            weighted_human_plan_prob_set = [weighted_human_plan_prob_set; [1:1:length(follower_plan_prob_set)]];
            weighted_human_plan_prob_set = weighted_human_plan_prob_set';
            weighted_human_plan_prob_set = sortrows(weighted_human_plan_prob_set,1);
            Delta = 0;
            acc_weighted_min = 100;
            acc_weighted_max = -100;
            yawrate_weighted_min = 100;
            yawrate_weighted_max = -100;
            for i = length(follower_plan_prob_set):-1:1
                
                human_acc_min = min(policy_set_human{weighted_human_plan_prob_set(i,2),1}(1,:));             
                human_acc_max = max(policy_set_human{weighted_human_plan_prob_set(i,2),1}(1,:));      
                human_yawrate_min = min(policy_set_human{weighted_human_plan_prob_set(i,2),1}(2,:));
                human_yawrate_max = max(policy_set_human{weighted_human_plan_prob_set(i,2),1}(2,:));
                acc_weighted_min = min(acc_weighted_min, human_acc_min);
                acc_weighted_max = max(acc_weighted_max, human_acc_max);
                yawrate_weighted_min = min(yawrate_weighted_min, human_yawrate_min);
                yawrate_weighted_max = max(yawrate_weighted_max, human_yawrate_max);
                Delta = Delta + weighted_human_plan_prob_set(i, 1);
                if Delta > ctrl_bound_Delta
                    break;
                end
            end
            human_acc_min_index = round((acc_weighted_min - grid.acc_grid(1))/(grid.acc_grid(2)-grid.acc_grid(1)) + 1);
            human_acc_min_index = min(max(human_acc_min_index, 1), length(grid.acc_grid));

            human_acc_max_index = round((acc_weighted_max - grid.acc_grid(1))/(grid.acc_grid(2)-grid.acc_grid(1)) + 1);
            human_acc_max_index = min(max(human_acc_max_index, 1), length(grid.acc_grid));
            
            human_yawrate_min_index = round((yawrate_weighted_min - grid.yawrate_grid(1))/(grid.yawrate_grid(2)-grid.yawrate_grid(1)) + 1);
            human_yawrate_min_index = min(max(human_yawrate_min_index, 1), length(grid.yawrate_grid));

            human_yawrate_max_index = round((yawrate_weighted_max - grid.yawrate_grid(1))/(grid.yawrate_grid(2)-grid.yawrate_grid(1)) + 1);
            human_yawrate_max_index = min(max(human_yawrate_max_index, 1), length(grid.yawrate_grid));
            
 
            % handle cases where min index and max index are the same
            if human_acc_min_index == human_acc_max_index
                if human_acc_min_index == 1
                    human_acc_max_index = human_acc_min_index + 1;
                else
                    human_acc_min_index = human_acc_min_index - 1;
                end
            end
            if human_yawrate_min_index == human_yawrate_max_index
                if human_yawrate_min_index == 1
                    human_yawrate_max_index = human_yawrate_min_index + 1;
                else
                    human_yawrate_min_index = human_yawrate_min_index - 1;
                end
            end
            our_BRT_id = BRT_map(strcat(num2str(human_yawrate_min_index), num2str(human_yawrate_max_index), num2str(human_acc_min_index), num2str(human_acc_max_index)));
            if our_BRT_id == 45
                our_BRT_id = 0;
            end
            if Bayesian_BRT_mode == "role"
                our_BRT_id = 6;
            end
            [issafe_ourBRT, safe_acc_ourBRT, safe_yawrate_ourBRT] = plot_BRT(rel5didx, s_w, human_car_index, nan, our_BRT_id, 'b', buffer_scale, "Our BRT", boundary_value,  '-.', 2, t);
            if ~issafe_ourBRT
                our_brt_violation = our_brt_violation + 1;
            end
            drawnow;
            set(gca,'XTick',[], 'YTick', [])
            legend('AutoUpdate','on','FontSize',14)
            for i = 1:1:length(plots_handle)
                delete(plots_handle{i});
            end
            
            % update the state 
            if log_sim_mode
                s_w(:,human_car_index) = getWorldState(human_car_index, t+1, traj);
                s_w(:,robot_car_index) = updateState(s_w(:,robot_car_index), policy_set_rob{best_boltzmann_plan_idx(1)}(:,1)+[ddd;0], env_params);
            else
                sw_human_temp = s_w(:,human_car_index)';
                sw_robot_temp = s_w(:,robot_car_index)';
                human_ctrl = policy_set_human{best_match, 1}(:,1);
                if human_mode == "aggressive"
                    s_w(:,human_car_index) = updateState(sw_human_temp, human_ctrl * 0 + [1;0], env_params);
                else
                    s_w(:,human_car_index) = updateState(sw_human_temp, human_ctrl, env_params);
                end
                sim_log{t,1}.human_ctrl = human_ctrl;
                if safety_mode == 0
                    robot_ctrl = policy_set_rob{best_boltzmann_plan_idx(1)}(:,1);
                elseif safety_mode == 1
%                     if ~issafe_full
%                         robot_ctrl = [safe_acc_full;safe_yawrate_full];    
%                     else
%                         robot_ctrl = policy_set_rob{best_boltzmann_plan_idx(1)}(:,1);
%                     end
                elseif safety_mode == 2
                    if ~issafe_ourBRT
                        robot_ctrl = [safe_acc_ourBRT;safe_yawrate_ourBRT];                    
                    else
                        robot_ctrl = policy_set_rob{best_boltzmann_plan_idx(1)}(:,1);
                    end
                end  
                s_w(:,robot_car_index) = updateState(s_w(:,robot_car_index), robot_ctrl, env_params);
                sim_log{t,1}.robot_ctrl = robot_ctrl; 
            end
            sim_log{t,1}.our_brt_violation = our_brt_violation;
            sim_log{t,1}.full_brt_violation = full_brt_violation;
        end
    end
end

function motion_plan = buildMotion(s_w,human_car_index, env_params, policy)
    motion_plan = [];
    sw_human_temp = s_w(:,human_car_index)';
    for k = 1:1:env_params.horizon
        sw_human_temp = updateState(sw_human_temp, policy(:,k), env_params);
        motion_plan = [motion_plan; sw_human_temp([1,2,3,5])];
    end
end
%% Extract action bound index given a acc policy
function [human_yawrate_min_index, human_yawrate_max_index, human_acc_min_index, human_acc_max_index] = get_action_bound_idx(human_acc_predict, human_yawrate_predict)
    global grid   
    acc_grid = grid.acc_grid;
    da = acc_grid(2) - acc_grid(1);
    
    yawrate_grid = grid.yawrate_grid;
    dy = yawrate_grid(2) - yawrate_grid(1);
    
    human_acc_min = min(human_acc_predict);
    human_acc_max = max(human_acc_predict);
    
    human_yawrate_min = min(human_yawrate_predict);
    human_yawrate_max = max(human_yawrate_predict);
   
    % find the cooresponding BRT id
    human_acc_min_index = round((human_acc_min - acc_grid(1))/da + 1);
    human_acc_min_index = min(max(human_acc_min_index, 1), length(acc_grid));

    human_acc_max_index = round((human_acc_max - acc_grid(1))/da + 1);
    human_acc_max_index = min(max(human_acc_max_index, 1), length(acc_grid));
    
    human_yawrate_min_index = round((human_yawrate_min - yawrate_grid(1))/dy + 1);
    human_yawrate_min_index = min(max(human_yawrate_min_index, 1), length(yawrate_grid));

    human_yawrate_max_index = round((human_yawrate_max - yawrate_grid(1))/dy + 1);
    human_yawrate_max_index = min(max(human_yawrate_max_index, 1), length(yawrate_grid));
    
    % handle cases where min index and max index are the same
    if human_acc_min_index == human_acc_max_index
        if human_acc_min_index == 1
            human_acc_max_index = human_acc_min_index + 1;
        else
            human_acc_min_index = human_acc_min_index - 1;
        end
    end
    
    if human_yawrate_min_index == human_yawrate_max_index
        if human_yawrate_min_index == 1
            human_yawrate_max_index = human_yawrate_min_index + 1;
        else
            human_yawrate_min_index = human_yawrate_min_index - 1;
        end
    end
end

%% Compute leader's policy
function leader_plan_prob_set = getLeaderPolicy(sw, leader_index, refPath_leader, follower_yaw_rate, env_params, policy_set, keys_follower, traj, t)
    global  Acc_sampled_policy_set
    
    R_set = zeros(1, size(policy_set,1));
    follower_idx = mod(leader_index, 2) + 1;
    residule_set = zeros(1, size(policy_set,1));
    follower_acc_policy_set = Acc_sampled_policy_set{keys_follower(1), 1};
    
    parfor leader_plan_index = 1:size(policy_set,1)
        
        leader_plan = policy_set{leader_plan_index, 1};
        R = 0;
        sw_temp = sw;
        ref_deviation = 0;
        for k = 1:1:env_params.horizon
            if leader_plan_index <= size(policy_set,1)
                sw_temp(:,leader_index) = updateState(sw_temp(:,leader_index)', leader_plan(:,k)', env_params)';
                s_sl = getStateFrenet(sw_temp(1:6,leader_index), refPath_leader);
                ref_deviation = ref_deviation + abs(s_sl(4));
                r = getStepReward(sw_temp, leader_index, env_params,'l');
            else
                s_w = zeros(7,2); % [x,y,theta,kappa,v,a,yaw_rate]
                s_w(:,follower_idx) = getWorldState(follower_idx, t + k, traj);
                s_w(:,leader_index) = getWorldState(leader_index, t + k, traj);
                r = getStepReward(s_w, leader_index, env_params,'l');
                
            end
            R = R + env_params.decay ^ (k-1) * r;
        end
        R = R - env_params.ref_weight * ref_deviation;
        R_set(leader_plan_index) = R;
        residule_set(leader_plan_index) = ref_deviation;
    end
    leader_plan_prob_set = zeros( length(env_params.beta_set), size(policy_set,1));
    for beta_idx = 1:1:length(env_params.beta_set)
        beta = env_params.beta_set(beta_idx);
        leader_plan_prob_set(beta_idx, :) = exp(beta * R_set) /  sum(exp(beta * R_set));
    end
end

%% Compute a human follower's policy distribution under different moel confidence parameter
function  follower_plan_prob_set = getBoltzmannPolicy(sw, follower_idx, leader_plan, refPath_follower, env_params, policy_set, traj, t, type)
    % save the raw reward of each plan
    R_set = zeros(1, size(policy_set,1));
    residule_set = zeros(1, size(policy_set,1));
    leader_index = mod(follower_idx, 2) + 1;
    parfor plan_index = 1:size(policy_set,1)
        R = 0;
        sw_temp = sw;
        ref_deviation = 0;
        for k = 1:1:env_params.horizon
            sw_temp(:,follower_idx) = updateState(sw_temp(:,follower_idx)', policy_set{plan_index}(:,k)', env_params)';
            s_sl = getStateFrenet(sw_temp(1:6,follower_idx), refPath_follower);
            ref_deviation = ref_deviation + abs(s_sl(4));
            r = getStepReward(sw_temp, follower_idx, env_params, type);
            R = R + env_params.decay ^ (k-1) * r;
        end
        if type == "r"
            R = R - env_params.ref_weight_r * ref_deviation;
        else
            R = R - env_params.ref_weight_ah * ref_deviation;
        end
        
        R_set(plan_index) = R;
        residule_set(plan_index) = ref_deviation;
    end
    follower_plan_prob_set = zeros( length(env_params.beta_set), size(policy_set,1));
    for beta_idx = 1:1:length(env_params.beta_set)
        beta = env_params.beta_set(beta_idx);
        follower_plan_prob_set(beta_idx, :) = exp(beta * R_set) /  sum(exp(beta * R_set));
    end
end

%% Compute a human follower's policy distribution under different moel confidence parameter
function  follower_plan_prob_set = getFollowerPolicy(sw, follower_idx, leader_plan, refPath_follower, env_params, policy_set, traj, t)
    R_set = zeros(1, size(policy_set,1));
    residule_set = zeros(1, size(policy_set,1));
    leader_index = mod(follower_idx, 2) + 1;
    parfor plan_index = 1:size(policy_set,1)
        R = 0;
        sw_temp = sw;
        ref_deviation = 0;
        for k = 1:1:env_params.horizon
            if plan_index <= size(policy_set,1)
                sw_temp(:,follower_idx) = updateState(sw_temp(:,follower_idx)', policy_set{plan_index}(:,k)', env_params)';
                s_sl = getStateFrenet(sw_temp(1:6,follower_idx), refPath_follower);
                ref_deviation = ref_deviation + abs(s_sl(4));
                sw_temp(:,leader_index) = updateState(sw_temp(:,leader_index)', leader_plan(:,k)', env_params)';
                r = getStepReward(sw_temp, follower_idx, env_params,'f');
            else
                s_w = zeros(7,2); % [x,y,theta,kappa,v,a,yaw_rate]
                s_w(:,follower_idx) = getWorldState(follower_idx, t + k, traj);
                s_w(:,leader_index) = getWorldState(leader_index, t + k, traj);
                r = getStepReward(s_w, follower_idx, env_params,'f');
            end
            R = R + env_params.decay ^ (k-1) * r;
        end
        R = R - env_params.ref_weight * ref_deviation;
        R_set(plan_index) = R;
        residule_set(plan_index) = ref_deviation;
    end
    follower_plan_prob_set = zeros( length(env_params.beta_set), size(policy_set,1));
    for beta_idx = 1:1:length(env_params.beta_set)
        beta = env_params.beta_set(beta_idx);
        follower_plan_prob_set(beta_idx, :) = exp(beta * R_set) /  sum(exp(beta * R_set));
    end
end 
function  follower_optimal_acc_plan = getFollowerAccPolicy(sw, follower_idx, leader_plan, follower_yaw_rate, env_params, policy_set)
    % save the raw reward of each plan
    R_set = zeros(1, size(policy_set,1));
    leader_index = mod(follower_idx, 2) + 1;
    % for each acc short horizon policy
    for plan_index = 1:size(policy_set,1)
        R = 0;
        sw_temp = sw;
        ref_deviation = 0;
        for k = 1:1:env_params.horizon
            sw_temp(:,follower_idx) = updateState(sw_temp(:,follower_idx)', [policy_set(plan_index, k), follower_yaw_rate(k)], env_params)';
            sw_temp(:,leader_index) = updateState(sw_temp(:,leader_index)', leader_plan(:,k)', env_params)';
            r = getStepReward(sw_temp, follower_idx, env_params,'f');
            R = R + env_params.decay ^ (k-1) * r;
        end
        R_set(plan_index) = R;
    end
    [~, best_plan_index] = max(R_set);
    follower_optimal_acc_plan = policy_set(best_plan_index, :);
end

%% Get world state of an agent from traj
function s_w = getWorldState(agent_id, t, traj)
    s_w = zeros(6,1); %[x,y,theta,kappa,v,a]
    s_w(1:3) = traj{agent_id,1}(1:3,t);
    if s_w(3) < 0
        s_w(3) = s_w(3) + 2 * pi;
    end
    s_w(4) = 1;
    s_w(5:7) = traj{agent_id,1}(4:6,t);
end

%% Extract traj from raw data
function traj = extract_traj(inter_id, t_start, duration)
    global data;
    traj = cell(2,1);
    % store the state trajectory
    s1_set = [];
    s2_set = [];
    % compute the frame offset
    frame_offset =  data{inter_id, 2}{1, 1} - data{inter_id, 3}{1, 1};
    
    for t_increment = t_start: 1 : t_start + duration
        if frame_offset < 0
            t_1 = t_increment - frame_offset;
            t_2 = t_increment;
        else
            t_2 = t_increment + frame_offset;
            t_1 = t_increment;
        end
        if t_1 > size(data{inter_id, 2}, 1) || t_2 > size(data{inter_id, 3}, 1)
            break;
        end
        s_1 = data{inter_id, 2}{t_1, 2};
        s_2 = data{inter_id, 3}{t_2, 2};
        if data{inter_id, 2}{t_1, 1} ~= data{inter_id, 3}{t_2, 1}
            disp("Frame wrong!");
            break;
        end
        s1_set = [s1_set, [s_1(1)+1000; s_1(2)+1000; s_1(5); sqrt(s_1(3)^2 + s_1(4)^2); 0; 0]];
        s2_set = [s2_set, [s_2(1)+1000; s_2(2)+1000; s_2(5); sqrt(s_2(3)^2 + s_2(4)^2); 0; 0]];
    end
    for i = 1:1:size(s1_set,2)-1
        s1_set(5,i) = (s1_set(4,i+1) - s1_set(4,i)) / 0.1; % dt in the data is 0.1
        s2_set(5,i) = (s2_set(4,i+1) - s2_set(4,i)) / 0.1;
        
        s1_set(6,i) = (s1_set(3,i+1) - s1_set(3,i)) / 0.1; % dt in the data is 0.1
        s2_set(6,i) = (s2_set(3,i+1) - s2_set(3,i)) / 0.1;
    end
    traj{1,1} = s1_set(:,1:end-1);
    traj{2,1} = s2_set(:,1:end-1);
end
function [rob_ref, human_ref] = getRefLane(traj, human_car_index)
    global plots_handle;
    waypoints = traj{human_car_index,1}(1:2,:)';
    human_ref = referencePathFrenet(waypoints,'DiscretizationDistance',0.01);
    connector = trajectoryGeneratorFrenet(human_ref);
    initState = [0,0,0,0.0,0,0];
    termState = [50 0 0 0.0 0 0];
    waypoints = traj{mod(human_car_index,2)+1,1}(1:2,:)';
    rob_ref = referencePathFrenet(waypoints,'DiscretizationDistance',0.01);
end

%% Generate a human car's trajectory prediction based on the current t_index
function prediction = getHumanPred(start_time, current_time, host_car, traj, s_w, terminal_point)
    global plots_handle;    
    waypoints = traj{host_car,1}(1:2,start_time : end)';
    % connect the waypoints to a terminal point 
    waypoints = [waypoints; terminal_point];
    refPath = referencePathFrenet(waypoints,'DiscretizationDistance',0.01);
    connector = trajectoryGeneratorFrenet(refPath);
    % assuming the human car accelerates, generate a prediction
    v_curr = traj{host_car,1}(4, current_time);
    a = 3;
    s_advanced = 0;
    for i = 1:1:5
        % advance 6 steps based on the reference
        v_curr = min(10, v_curr + 0.5 * a);
        s_advanced = s_advanced + v_curr * 0.5;
    end
    % initial state of the human car
    s_sl = getStateFrenet(s_w(1:6,host_car), refPath);
    initState = [s_sl(1),0,0,0,0,0];
    termState = [initState(1)+s_advanced 0 0 0 0 0];
    % deviate from the true location
    termState(4) = 1;
    [~,prediction] = connect(connector,initState,termState, 8);
    prediction = [prediction.Trajectory(:,1),prediction.Trajectory(:,2)];
    %figure(1)
    plots_handle{end + 1} = plot(prediction(:,1),prediction(:,2),'r','LineWidth',2,'LineStyle','-','DisplayName','Frenet reference');
    prediction = [ waypoints(start_time:current_time-1,:);prediction];
end

%% Compute game state in human car's s-l frame
% Input: s_w  current game state world frame, human car's (long) prediction [x1 y1; x2 y2; ...] serves as reference
function s_sl = getStateFrenet(s_w, refPath)
    s_sl = [];
    % convert the world state to s-l frame
    for agent_id = 1:1:size(s_w, 2)
        while true
            try
                frenetState = global2frenet(refPath,s_w(:, agent_id)');
                break;
            catch
                s_w(3, agent_id) = s_w(3, agent_id) - 0.01; 
            end
        end
        s_sl = [s_sl, frenetState'];
    end
end


%% Compute relative state index
function rel5didx = getRelativeStateIndex(s_w, human_car_idx)
    global grid
    v_h = s_w(5,human_car_idx) + 0.01;
    v_r = s_w(5,mod(human_car_idx,2)+1) + 0.01;
    v_h_idx = round( (v_h - grid.v_grid(1))/(grid.v_grid(2) - grid.v_grid(1)) + 1);
    v_r_idx = round( (v_r - grid.v_grid(1))/(grid.v_grid(2) - grid.v_grid(1)) + 1);
    psi_rel = s_w(3,human_car_idx) - s_w(3,mod(human_car_idx,2)+1);
    psi_rel = psi_rel / pi * 180;
    if psi_rel > 180
        psi_rel = psi_rel - 360;
    end
    if psi_rel < -180
        psi_rel = psi_rel + 360;
    end
    phi_rel_idx = round((psi_rel - grid.phi_grid(1))/(grid.phi_grid(2) - grid.phi_grid(1)) + 1);
    rel5didx = [v_h_idx, v_r_idx, phi_rel_idx];
end


%% plot the BRT functions
function [issafe, safe_acc, safe_yawrate] = plot_BRT(rel5didx, s_w, human_car_index, refPath_human, BRT_id, color, buffer, BRT_name, boundary_value, BRT_line_style, BRT_line_width, t)
    global plots_handle show_frenet_BRT sim_log ;
    if show_frenet_BRT
        subplot(1, 2, 1);
    end
    %hold on;
    %axis equal;
    if BRT_name == "True BRT"
        istrue = 1;
    else
        istrue = 0;
    end
    [BRT_w, BRT_local, issafe, rel_state, safe_acc, safe_yawrate] = get_BRT(rel5didx, s_w, human_car_index, nan, BRT_id, buffer, boundary_value, istrue);
    if BRT_name == "True BRT"
        sim_log{t,1}.true_brt_w = BRT_w;
        %sim_log{t,1}.true_brt_sl = BRT_sl;
    elseif BRT_name == "Our BRT"
        sim_log{t,1}.our_brt_w = BRT_w;
        %sim_log{t,1}.our_brt_sl = BRT_sl;
    elseif BRT_name == "Full BRT"
        sim_log{t,1}.full_brt_w = BRT_w;
        %sim_log{t,1}.full_brt_sl = BRT_sl;
    end
    
    
    if ~isempty(BRT_w)
        plots_handle{end + 1} = plot(BRT_w(1,:),BRT_w(2,:), color,'LineWidth',BRT_line_width, 'DisplayName', BRT_name,'LineStyle', BRT_line_style);
    end
    if show_frenet_BRT
        subplot(1,2,2)
    
        hold on;
        axis([-20, 20, -20, 20])
        if ~isempty(BRT_w)
            plots_handle{end + 1} = plot(BRT_local(1,:),BRT_local(2,:),color,'LineWidth',3);
        end
        plots_handle{end + 1} = plot_car([0,0,0], "car-orange.png", 1, 5.5, 2.5);
        
        rob_index =  mod(human_car_index, 2) + 1;
       
        psi_rel = s_w(3,human_car_index) - s_w(3,mod(human_car_index,2)+1);
        psi_rel = psi_rel / pi * 180;
        if psi_rel > 180
            psi_rel = psi_rel - 360;
        end
        if psi_rel < -180
            psi_rel = psi_rel + 360;
        end 
        plots_handle{end + 1} = plot_car([rel_state(1),rel_state(2),psi_rel], "car-red.png", 1, 5.5, 2.5);
    end
end


%% Compute full BRT
function [ BRT_w, BRT_local, issafe, rel_state, safe_acc, safe_yawrate] = get_BRT(rel5didx, s_w, human_car_idx, refPath, BRT_id, buffer, boundary_value, istrue)
    global BRT_all grid BRT_all_true env_params safe_yawrate_all safe_acc_all BRT_all_2c safe_yawrate_all_2c safe_acc_all_2c
    issafe = 1;
    safe_status = 1;
    safe_acc = nan;
    safe_yawrate = nan;
    % get the BRT boundary in the local frame
    v_h_idx = rel5didx(1);
    v_r_idx = rel5didx(2);
    phi_rel_index = rel5didx(3);
    if BRT_id == 0
        BRT_data = readNPY("BRS/June_14/reldyn5d_brs_mode" + num2str(BRT_id) + ".npy");
        BRT_2d_full = BRT_data(:,:,phi_rel_index, v_h_idx, v_r_idx);
    else
        BRT_data = readNPY("BRS/Aug_26/reldyn5d_brs_mode" + num2str(BRT_id) + ".npy");
        BRT_2d_full = BRT_data(:,:,phi_rel_index, v_h_idx, v_r_idx);
    end
    x_max = 20;
    x_min = -20;
    x_pts = 81;
    y_max = 20;
    y_min = -20;
    y_pts = 81;
    d_x = (x_max - x_min) / (x_pts - 1);
    d_y = (y_max - y_min) / (y_pts - 1);
    x_grid = x_min : d_x : x_max;
    y_grid = y_min : d_y : y_max;
    [X, Y] = meshgrid(x_grid, y_grid);
    [M, c] = contour(X, Y, BRT_2d_full',[boundary_value boundary_value],'LineColor','r','LineWidth', 1);
    delete(c);
    if isempty(M)
        max_v = max(max(BRT_2d_full));
        [M, c] = contour(X, Y, BRT_2d_full',[max_v max_v],'LineColor','r','LineWidth', 1);
        delete(c);
        issafe = 0;
    end
    BRT_local = [M(:,2:end)];
    
    % convert boundary to the world frame
    dx = s_w(1,mod(human_car_idx,2)+1) - 0;
    dy = s_w(2,mod(human_car_idx,2)+1) - 0;
    psi_rel = grid.phi_grid(phi_rel_index); 
    d_theta = s_w(3,mod(human_car_idx,2)+1)/pi*180;
    if d_theta < 0
        d_theta = d_theta + 360;
    end
    H = [cosd(d_theta) -sind(d_theta) dx;
         sind(d_theta) cosd(d_theta) dy;
         0 0 1];
    BRT_w = [];
    
    for i = 2:1:size(M,2)
        % convert to human car's frenet frame
        BRT_point = H * [M(:,i)*buffer; 1];
        BRT_w = [BRT_w BRT_point(1:2)];
    end
    
    rel_state = inv(H) * [s_w(1,human_car_idx);s_w(2,human_car_idx);1 ];
    rel_state(3) = grid.phi_grid(phi_rel_index);
    
    x_ego = s_w(1,human_car_idx);
    y_ego = s_w(2,human_car_idx);
    theta_ego = s_w(3,human_car_idx);
    l_car = env_params.l_car;
    w_car = env_params.w_car;
    
    l_car_safe = l_car;    
    w_car_safe = w_car;
    
    Ego_rectangle = [x_ego-l_car_safe/2*cos(theta_ego)+w_car_safe/2*sin(theta_ego), y_ego-l_car_safe/2*sin(theta_ego)-w_car_safe/2*cos(theta_ego);
        x_ego-l_car_safe/2*cos(theta_ego)-w_car_safe/2*sin(theta_ego), y_ego-l_car_safe/2*sin(theta_ego)+w_car_safe/2*cos(theta_ego);
        x_ego+l_car_safe/2*cos(theta_ego)-w_car_safe/2*sin(theta_ego), y_ego+l_car_safe/2*sin(theta_ego)+w_car_safe/2*cos(theta_ego);
        x_ego+l_car_safe/2*cos(theta_ego)+w_car_safe/2*sin(theta_ego), y_ego+l_car_safe/2*sin(theta_ego)-w_car_safe/2*cos(theta_ego)];
    issafe = ~isintersect(Ego_rectangle', BRT_w);
    for dalpha = 0:0.01:2*pi
        x_t = rel_state(1) + env_params.l_car * cos(dalpha) * buffer*1;
        y_t = rel_state(2) + env_params.l_car * sin(dalpha) * buffer*1;
        x_t_index =  max(min( round((x_t + 20) * 81/40 + 1), 81), 1);
        y_t_index = max(min(round((y_t + 20) * 81/40 + 1), 81),1);
        if BRT_2d_full(x_t_index,y_t_index) < boundary_value
            if BRT_id == 0
                date_ = "June_14";
            else
                date_ = "Aug_26";
            end
            safe_acc_data = readNPY("BRS/"+date_+"/reldyn5d_ctrl_acc_mode" + num2str(BRT_id) + ".npy");
            safe_acc_2d = safe_acc_data(:,:,phi_rel_index, v_h_idx, v_r_idx);
            safe_acc = safe_acc_2d(x_t_index,y_t_index);
            safe_yawrate_data = readNPY("BRS/"+date_+"/reldyn5d_ctrl_beta_mode" + num2str(BRT_id) + ".npy");
            safe_yawrate_2d = safe_yawrate_data(:,:,phi_rel_index, v_h_idx, v_r_idx);
            safe_yawrate = safe_yawrate_2d(x_t_index,y_t_index) / env_params.dt;
            
            break;
        end
        
    end
end

%% Compute the reward of the current game state (this reward does not reflect the jerk)
function R = getStepReward(s_w, car_index, env_params, flag)
    % get the feature
    feature = get_feature(s_w, car_index, env_params); 
    % compute the linear reward
    if flag == "l"
        R = env_params.l_weights * feature;
    elseif flag == "f"
        R = env_params.f_weights * feature;
    elseif flag == "r"
        R = env_params.r_weights * feature;
    elseif flag == "ah"
        R = env_params.ah_weights * feature;
    end
    
end

%% Update the state of a vehicle
function s_w_new = updateState(s_w_car, u, env_params)
    %%% s_w_car: a car's world state [x,y,theta,kappa,v,a]
    %%% u: input acc control and yaw control
    s_w_new = s_w_car;
    % update speed
    s_w_new(5) = min(max(s_w_car(5) + u(1) * env_params.dt, 0.1), env_params.v_max);
    s_w_new(6) = u(1);
    % update yaw angle
    s_w_new(3) = s_w_car(3) + u(2) * env_params.dt;
    % update the position
    s_w_new(1) = s_w_car(1) + s_w_new(5) * cos(s_w_new(3)) * env_params.dt;
    s_w_new(2) = s_w_car(2) + s_w_new(5) * sin(s_w_new(3)) * env_params.dt;
end

function pathlength = getPathLength(path)
    x = path.Waypoints(:,1);
    y = path.Waypoints(:,2);
    pathlength = 0;
    for i = 2:1:size(x, 1)
        pathlength = pathlength + sqrt((x(i) - x(i-1))^2 + (y(i) - y(i-1))^2);
    end
end
