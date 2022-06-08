%% Compute the feature
% f1 - distance to the robot car
% f2 - speed
% f3 - collision
% s_w = [x1 y1 theta1 kappa1 v1 a1]' world state
function feature = get_feature(s_w, car_index, env_params)
 
    % initialize the feature
    feature = zeros(6,1);
  
    % feature 1 distance between two cars
    d = sqrt((s_w(1,1) - s_w(1,2))^2 + (s_w(2,1)-s_w(2,2))^2);
    d = min(d, 60) / (60); % clip the maximum distance to be 3 car length
    
    %feature(1) = - exp(- d / 10);
    feature(1) = - 1 / (d^2 + 0.1);
       
    % feature 2: speed
    feature(2) =  - 1 / (s_w(5, car_index) / env_params.v_max + 0.1);
    
    %feature(2) =  - exp(- s_w(5, car_index) / 10);
    
    % feature 3: stop
    if s_w(5, car_index) == 0
        feature(3) = -1;
    end
    
    % feature 4: safety
    feature(4) = -check_if_collision(s_w, env_params, 4);
    
    % feature 5: collision
    feature(5) = -check_if_collision(s_w, env_params, 1);
    
    if s_w(6, car_index) > 0
        feature(6) = -abs(s_w(6, car_index));
    end
    
end
