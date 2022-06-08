function if_collision = check_if_collision(s_w,env_params, scale)
    
    l_car = env_params.l_car;
    w_car = env_params.w_car;
    
    l_car_safe = 1.0 * l_car * scale;    
    w_car_safe = 1.0 * w_car * scale;
    
    x_ego = s_w(1,1);
    y_ego = s_w(2,1);
    theta_ego = s_w(3,1);
    
    Ego_rectangle = [x_ego-l_car_safe/2*cos(theta_ego)+w_car_safe/2*sin(theta_ego), y_ego-l_car_safe/2*sin(theta_ego)-w_car_safe/2*cos(theta_ego);
        x_ego-l_car_safe/2*cos(theta_ego)-w_car_safe/2*sin(theta_ego), y_ego-l_car_safe/2*sin(theta_ego)+w_car_safe/2*cos(theta_ego);
        x_ego+l_car_safe/2*cos(theta_ego)-w_car_safe/2*sin(theta_ego), y_ego+l_car_safe/2*sin(theta_ego)+w_car_safe/2*cos(theta_ego);
        x_ego+l_car_safe/2*cos(theta_ego)+w_car_safe/2*sin(theta_ego), y_ego+l_car_safe/2*sin(theta_ego)-w_car_safe/2*cos(theta_ego)];
    
    x_opp = s_w(1,2);
    y_opp = s_w(2,2);
    theta_opp = s_w(3,2);
    
    opp_rectangle = [x_opp-l_car_safe/2*cos(theta_opp)+w_car_safe/2*sin(theta_opp), y_opp-l_car_safe/2*sin(theta_opp)-w_car_safe/2*cos(theta_opp);
            x_opp-l_car_safe/2*cos(theta_opp)-w_car_safe/2*sin(theta_opp), y_opp-l_car_safe/2*sin(theta_opp)+w_car_safe/2*cos(theta_opp);
            x_opp+l_car_safe/2*cos(theta_opp)-w_car_safe/2*sin(theta_opp), y_opp+l_car_safe/2*sin(theta_opp)+w_car_safe/2*cos(theta_opp);
            x_opp+l_car_safe/2*cos(theta_opp)+w_car_safe/2*sin(theta_opp), y_opp+l_car_safe/2*sin(theta_opp)-w_car_safe/2*cos(theta_opp)];        
   % full collision check
   if_collision = isintersect(Ego_rectangle', opp_rectangle');
   %if_collision = simple_collision_check(Ego_rectangle(2,:), Ego_rectangle(4,:), opp_rectangle(2,:), opp_rectangle(4,:));
    
end