% This code is used to check collision when the box is alligned with the
% x(y) axis
function is_collision = simple_collision_check(l1, r1, l2, r2)
    is_collision =  1;
    % If one rectangle is on left side of other 
    if (l1(1) >= r2(1) || l2(1) >= r1(1)) 
        is_collision =  0;
    end
    % If one rectangle is above other 
    if (l1(2) <= r2(2) || l2(2) <= r1(2)) 
        is_collision =  0;
    end
end