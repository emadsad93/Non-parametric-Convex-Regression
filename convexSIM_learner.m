function [theta, y_hat, beta_hat, w_hat] = convexSIM_learner(x, y, w_star, numIter, theta_star, y_true, show_plots, w_init)

%%
run(['cvx' filesep 'cvx_setup'])

if (numIter <= 0)
    fprintf('Number of iterations should be positive.\n')
    return;
end

[d, n] = size(x);

%% For testing stuff

numIter = 1500; 
y_error = zeros(numIter, 1);
y_hat_copy = zeros(n, numIter);
theta_copy = zeros(n, numIter);
beta_hat_copy = zeros(n, numIter);
what_copy = zeros(d, numIter);

%% Algorithm

w_old = w_init; 

func = @(theta)(theta);
%func =@(theta)(0);


%y_hat = func(w_old' * x);

%for t = 1:numIter
iter = 1; 
tol = 1e-6; 
change = realmax; 
y_hat = zeros(1,length(y)); 
error_new = sum((y - y_hat).^2); 
MSE = Inf; 

while((abs(change) > tol)&&(iter < 1500))
   
    y_hat_prev = y_hat;
    error_prev = error_new; 
    % Projections of the data onto w'x hyperplane
    what_copy(:, iter) = w_old;
    
    theta = w_old' * x;
    
    % QP program
    [y_hat, beta_hat]= QP_reg(theta, y, theta_star, y_true, show_plots);
    delta_y = y - y_hat;
    w_old = w_old + (x * delta_y')/n;%/n;
    w_old = w_old/norm(w_old);
    error_new = sum((y - y_hat).^2); 
    change =  (error_new - error_prev)/error_prev; 

    % Save empirical error y_hat and theta in each iteration.
    if mod(iter,10)==0
        fprintf('iter = %d,\t MSE = %1.4g,\t change = %1.2g\n',...
            iter,mean((y_hat(:)-y(:)).^2),change);
    end
%     fprintf('iter = %d, MSE = %f,\t change is is %f... y_hat is %f...\n',t, mean((y_hat(:) - y(:)).^2), norm(y), norm(y_hat));
    y_error(iter) = mean((y_hat(:) - y(:)).^2); 
    MSE = y_error(iter); 
    theta_copy(:, iter) = theta;
    beta_hat_copy(:, iter) = beta_hat;
    y_hat_copy(:, iter) = y_hat;
    iter = iter + 1;
    if (MSE < 10*tol)
        break;
    end
end

%save('empirical_error.mat', 'y_error');
ind = 1:(iter-1); 
[~, ind_new] = min(y_error(ind));
fprintf('Minimum MSE was at iter = %d\n', ind_new);

y_hat = y_hat_copy(:, ind_new);
theta = theta_copy(:, ind_new);
beta_hat = beta_hat_copy(:, ind_new);
w_hat = what_copy(:, ind_new); 


%keyboard;
end



