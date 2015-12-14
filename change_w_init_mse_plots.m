%% Clear Workspace
clear; close all; clc;

%% Training data
n = 400; d = 2;
x = zeros(d,n);
num_train = 200;

for i = 1:n
    x(:,i) = randn(d,1);
    x(:,i) = x(:,i)/(norm(x(:,i)));
end

x_train = x(:,1:num_train);
x_test = x(:,(num_train+1):n);

cvx_func = @(theta)(2 * theta.^2 - 1);
g_star = cvx_func;

w_star = randn(d,1);
w_star = w_star/norm(w_star);
y_true = g_star(w_star' * x_train);
theta_star = w_star' * x_train;

%% Initializations
a = 1; b = 5;
mse_matrix_1 = zeros(a,b);
mse_matrix_2 = zeros(a,b);
mse_matrix_3 = zeros(a,b);

show_plots = 0;

numIter = 1500;

%% Test Data

theta_test = w_star' * x_test;
y_test = g_star(theta_test);
[~, ind_test] = sort(theta_test);
theta_test = theta_test(ind_test); y_test = y_test(ind_test);

%%
sigma = 0.1;
x = x_train;
n = num_train;
eta = linspace(0,1,5);
for j = 1:b
    noise = randn(1,n);
    y = y_true + (sigma * (noise));
    y_test_noise = y_test + (sigma * (noise));
    for k = 1:length(eta)
        % Initialize w
        w_init = w_star + randn(d,1) * eta(k);
        %w_init =  2 * (rand(d,1) -1/2); %-rand(d,1);
        % Method 1: used QP and perceptron update
        fprintf('1) %d, %d\n', j,k);
        
        [theta, y_hat, beta_hat, w_hat] = convexSIM_learner(x,y,w_star, numIter, theta_star, y_true, show_plots, w_init);
        [~, ind] = sort(theta);
        theta_1 = theta(ind); y_hat_1 = y_hat(ind); beta_hat_1 = beta_hat(ind);
        w_hat_1 = w_hat;
        
        [~, y_hat_1] = interp_procedure(w_hat_1' * x_test, y_test_noise, y_hat_1,theta_1, beta_hat_1);
        % Method 2: ignored g_star, just used perceptron
        fprintf('2) %d, %d\n', j,k);
        
        [theta, y_hat, w_hat] = ignoreGstar(x,y,w_star, numIter, theta_star, y_true, show_plots, w_init);
        
        w_hat_2 = w_hat;
        
        theta_test_2 = w_hat_2' * x_test;
        y_hat_2 = theta_test_2;
        
        
        % Method 3: known g_star, best case scenario
        fprintf('3) %d, %d\n', j,k);
        
        [theta, y_hat, w_hat] = knownGstar(x,y,w_star, numIter, theta_star, y_true, show_plots, g_star, w_init);
        
        w_hat_3 = w_hat;
        
        theta_test_3 = w_hat_3' * x_test;
        
        y_hat_3 = g_star(theta_test_3);
        
        % Testing
        mse_matrix_1(j,k) = mean((y_hat_1 - y_test_noise').^2);
        fprintf('%d\n', mse_matrix_1(j,k));
        mse_matrix_2(j,k) = mean((y_hat_2 - y_test_noise).^2);
        fprintf( '%d\n',mse_matrix_2(j,k));
        mse_matrix_3(j,k) = mean((y_hat_3 - y_test_noise).^2);
        fprintf( '%d\n',mse_matrix_3(j,k));
        
        
        cvx_clear;
    end
    
end




