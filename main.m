%% Generate random data 
clear; close all; clc;
n = 200; d = 2; 
x = zeros(d,n);
for i = 1:n
    x(:,i) = randn(d,1);
    x(:,i) = x(:,i)/(norm(x(:,i)));
end

%% No noise %% 
%%%%%%%%%%%%%%
%cvx_func = @(theta)(2 * theta.^4 - 1);
%cvx_func = @(theta)(exp(-1 * theta/2));
%cvx_func = @(theta)(2 * theta.^8 - 1);
cvx_func = @(theta)(2 * theta.^2 - 1);
%cvx_func = @(theta)(exp(theta * -1/2)); 
%cvx_func = @(theta)(1./(1 +3 *exp(theta))); 
%cvx_func = @(theta)(theta);
%cvx_func = @(theta)(abs(theta)); 
%cvx_func = @(theta)(-theta .* abs(log(theta)));
%cvx_func = @(theta)(abs(log(theta)));
%cvx_func = @(theta)(5 * theta + 102);

sparsity_level = 0; 
noise_level = 0; 
[d,n] = size(x); 
g_star = cvx_func; 

if (sparsity_level == 0)
    w_star = randn(d,1);
    w_star = w_star/norm(w_star);
    y_true = g_star(w_star' * x); 
    noise = randn(1,n); 
    y = g_star(w_star' * x) + (noise_level * (noise));%/norm(noise)); 
else
    if (n < d) 
        fprintf('d must be >> n in the high dimensional setting\n'); 
    end
    s_idx = randi(d,sparsity_level,1);
    s_value = normrnd(0,1,sparsity_level,1);
    w_star = zeros(d,1);    
    w_star(s_idx) = s_value; 
    w_star = w_star/norm(w_star);
    y_true = g_star(w_star' * x); 
    y = g_star(w_star' * x) + (noise_level * randn(1,n));
end

save('forTesting', 'x', 'y', 'w_star'); 

theta_star = w_star' * x; 
show_plots = 1; 
numIter = 100; 

%% w_init
% w_old = w_star; 
w_init = 2 * (rand(d,1) -1/2); %

[theta, y_hat, beta_hat, w_hat] = convexSIM_learner(x,y,w_star, numIter, theta_star, y_true, show_plots, w_init); 

[~, ind] = sort(theta); 
theta = theta(ind); y_hat = y_hat(ind); beta_hat = beta_hat(ind); w_hat = w_hat; 

%% Plotting 
figure(13)
xlabel('$\theta$','Interpreter','LaTex', 'FontSize',20)
ylabel('$\hat{y}$','Interpreter','LaTex', 'FontSize',20)
hold on 
[~, ind2] = sort(theta_star); 
scatter(theta_star(ind2), y(ind2),40,'m', 'LineWidth',1.5)

plot(theta_star(ind2), y_true(ind2), 'g', 'LineWidth',1.5)
scatter(theta, y_hat, 40,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
plot(theta, y_hat, 'b', 'LineWidth',2)
axis('tight')

% beta_hat(1) = beta_hat(2);
% n_b = length(beta_hat);
% beta_hat(n_b) = beta_hat(n_b-1);
% for k = 4:(length(beta_hat)-4)
%     tt = linspace(theta(k) - 0.1, theta(k) + 0.1, 100); 
%     tangent_line = (beta_hat(k) * (tt - theta(k))) + y_hat(k); 
%     plot(tt, tangent_line, '--r', 'LineWidth',1.5); 
%     drawnow
% end 
% legend('noisy labels', 'true function', 'predicted labels', 'predicted function', 'subgradients')
legend('noisy labels', 'true function', 'predicted labels', 'predicted function')

hold off; 
    
%% Testing 
 
k = 2 * n; 
test = zeros(d,k);

for i = 1:k 
    test(:,i) = randn(d,1);
    test(:,i) = test(:,i)/(norm(test(:,i)));   
end 

testData = w_star' * test; 

y_test = g_star(testData);

[test_inter_err, y_inter] = interp_procedure(testData, y_test, y_hat,theta, beta_hat);
figure(19)
plot(theta, y_hat, 'go', testData, y_inter, 'ro');
fprintf('Method 1 is "Max of All Subgradient Inequalities"\n'); 
fprintf('Empirical error for method 1 is %f\n', norm(y_test' - y_inter)); 
title('Used "Maximum of Subgradient Inequalities" method to Predict the Labels for Test Data')
axis('tight');
legend('training', 'testing')

figure(20)
y_t = interp1(theta,y_hat,testData,'linear');
plot(theta, y_hat, 'go', testData, y_t, 'ro');
fprintf('Method 2 is "Linear Interpolation"\n'); 
fprintf('Empirical error for method 2 is %f\n', norm(y_test - y_t)); 
title('Used Linear Interpolation on Training Data to Predict the Labels for Test Data')
axis('tight');
legend('training', 'testing')

