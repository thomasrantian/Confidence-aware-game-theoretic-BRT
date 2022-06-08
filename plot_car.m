function handleIM = plot_car(car_state, car_file, trans_scale,l_car, w_car)
    car_file = "car_figs/" + car_file;
    x_c = car_state(1);
    y_c = car_state(2);
    theta = car_state(3);
    [marker,map,transperancy ] = imread(car_file);
    marker_2 = imrotate(marker,theta);
    transperancy_2 = imrotate(transperancy,theta) * trans_scale;
    handleIM = 0;
    if theta > 360
        theta = theta - 360;
    end
    if theta < 0
        theta = theta + 360;
    end
    if  theta >= 0 && theta <= 90
        
        x_l = x_c - l_car/2 * cos(theta / 180 * pi) - w_car/2 * sin(theta / 180 * pi);
        y_l = y_c - l_car/2 * sin(theta / 180 * pi) - w_car/2 * cos(theta / 180 * pi);
        x_h = x_c + l_car/2 * cos(theta / 180 * pi) + w_car/2 * sin(theta / 180 * pi);
        y_h = y_c + l_car/2 * sin(theta / 180 * pi) + w_car/2 * cos(theta / 180 * pi);
        handleIM = imagesc([x_l x_h], [y_h y_l], marker_2);
    
    elseif theta > 90 && theta <= 180
        theta = 180 - theta;
      
        x_l = x_c - l_car/2*cos(theta/180*pi)-w_car/2 *sin(theta/180*pi);
        x_h = x_c + l_car/2*cos(theta/180*pi)+w_car/2 *sin(theta/180*pi);
        y_l = y_c - l_car/2*sin(theta/180*pi)-w_car/2 *cos(theta/180*pi);
        y_h = y_c + l_car/2*sin(theta/180*pi)+w_car/2 *cos(theta/180*pi);
        handleIM = imagesc([x_l x_h], [y_h y_l], marker_2);
    
    elseif theta > 180 && theta <= 270
        theta = 270 - theta;
        x_l = x_c - w_car/2 *cos(theta/180*pi)-l_car/2*sin(theta/180*pi);
        x_h = x_c + w_car/2 *cos(theta/180*pi)+l_car/2*sin(theta/180*pi);
        y_l = y_c + l_car/2*cos(theta/180*pi)+w_car/2 *sin(theta/180*pi);
        y_h = y_c - l_car/2*cos(theta/180*pi)-w_car/2 *sin(theta/180*pi);
        handleIM = imagesc([x_l x_h], [y_l y_h], marker_2);
    
    elseif  theta >= 270 
        theta = theta - 270;
        x_l = x_c - w_car/2 * cos(theta / 180 * pi) - l_car/2 * sin(theta / 180 * pi);
        x_h = x_c + w_car/2 * cos(theta / 180 * pi) + l_car/2 * sin(theta / 180 * pi);
        
        y_l = y_c + l_car/2 * cos(theta / 180 * pi) + w_car/2 * sin(theta / 180 * pi);
        y_h = y_c - l_car/2 * cos(theta / 180 * pi) - w_car/2 * sin(theta / 180 * pi);
        handleIM = imagesc([x_l x_h], [y_l y_h], marker_2);
    end
    
    set(handleIM ,'AlphaData',transperancy_2);
end
