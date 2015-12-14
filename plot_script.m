clear; close all; clc;

%%%%%%%%%
% Training data
n = 200; d = 2;
x = zeros(d,n);
for i = 1:n
    x(:,i) = randn(d,1);
    x(:,i) = x(:,i)/(norm(x(:,i)));
end

cvx_func = @(theta)(2 * theta.^2 - 1);

a = 5; b = 5;
mse_matrix = zeros(a,b);
mse_matrix1 = zeros(a,b);
mse_matrix2 = zeros(a,b);


g_star = cvx_func;

w_star = randn(d,1);
w_star = w_star/norm(w_star);
y_true = g_star(w_star' * x);
numIter = 200;
theta_star = w_star' * x;
show_plots = 0;

%%%%%%%%%
% Test Data

k = 2 * n;
test = zeros(d,k);

for i = 1:k
    test(:,i) = randn(d,1);
    test(:,i) = test(:,i)/(norm(test(:,i)));
end

testData = w_star' * test;

y_test = g_star(testData); 
%%%%%%%%%
sigma = linspace(0,1,a);
for j = 1: b
    noise = randn(1,n);
    noise1 = randn(1, 2*n); 
    for k = 1:length(sigma)
        y = y_true + (sigma(k) * (noise));
        test_noise = y_test + (sigma(k) * (noise1));
        [theta, y_hat, beta_hat] = convexSIM_learner(x,y,w_star, numIter, theta_star, y_true, show_plots);
        
        [~, ind] = sort(theta);
        theta = theta(ind); y_hat = y_hat(ind); beta_hat = beta_hat(ind);
        
        % Testing
        [~, y_inter] = interp_procedure(testData, test_noise, y_hat,theta, beta_hat);
        mse_matrix(j,k) = norm(y_inter - y_test');
        mse_matrix1(j,k) = norm(y_test - test_noise);
        mse_matrix2(j,k) = norm(test_noise - testData);
        
        cvx_clear;
    end
    
end




