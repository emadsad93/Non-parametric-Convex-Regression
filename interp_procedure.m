function [test_inter_err, y_inter] = interp_procedure(testData, y_test, y_hat,theta, beta_hat)
    beta_hat(1) = beta_hat(2); 
    n_b = length(beta_hat); 
    beta_hat(n_b) = beta_hat(n_b-1); 
    n_test = length(y_test); 
    n = length(y_hat); 
    test_inter_err = zeros(n_test,1); 
    y_inter = zeros(n_test, 1); 

    for i = 1:n_test
        test_theta = repmat(testData(i), n,1); 
        temp = y_hat + (beta_hat .* (test_theta - theta)); 
        y_inter(i) = max(temp); 
        test_inter_err(i) = (y_test(i) - y_inter(i)); 
    end     

end 
