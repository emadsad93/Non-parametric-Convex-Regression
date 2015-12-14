function [y_hat, beta_hat] = QP_reg(theta, y, theta_star, y_true, show_plots)

n = length(y);
[theta,idx] = sort(theta,'ascend');
y1 = y;
y = y(idx);

cvx_begin quiet
variable y_hat(1, n);
variable beta_hat(1, n);
minimize(sum_square(y - y_hat));
subject to
for i = 1:(n-1)
    beta_hat(i+1) >= beta_hat(i);
    y_hat(i+1) >= (y_hat(i) + (theta(i+1) - theta(i)) * beta_hat(i));
    y_hat(i) >= (y_hat(i+1) + (theta(i)   - theta(i+1)) * beta_hat(i+1));
end
cvx_end



if (show_plots == 1)
    figure(12)
    clf;
    xlabel('$\theta$','Interpreter','LaTex', 'FontSize',24)
    ylabel('$\hat{y}$','Interpreter','LaTex', 'FontSize',24)
    
    hold on
    [~, ind2] = sort(theta_star);
    scatter(theta_star(ind2), y1(ind2),'full','MarkerFaceColor',[1 .5 0],'LineWidth',0.5)
    
    plot(theta_star(ind2), y_true(ind2),'g' ,'LineWidth',1.5)
    scatter(theta, y_hat, 40,'MarkerEdgeColor',[0 .5 .5],...
        'MarkerFaceColor',[0 .7 .7],...
        'LineWidth',1.5)
    plot(-theta, y_hat, 'b', 'LineWidth',2)
    axis('tight')
    
    beta_hat(1) = beta_hat(2);
    n_b = length(beta_hat);
    beta_hat(n_b) = beta_hat(n_b-1);
    for k = 4:(length(beta_hat)-4)
        tt = linspace(theta(k) - 0.1, theta(k) + 0.1, 100);
        tangent_line = (beta_hat(k) * (tt - theta(k))) + y_hat(k);
        plot(tt, tangent_line, '--r', 'LineWidth',1.5);
        drawnow
    end
    legend('noisy labels', 'true function', 'predicted labels', 'predicted function', 'subgradients')
    hold off;

end

theta(idx) = theta;
y_hat(idx) = y_hat;

beta_hat(idx) = beta_hat;


end


