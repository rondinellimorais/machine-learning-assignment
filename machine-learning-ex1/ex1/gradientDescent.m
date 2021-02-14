function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  %   taking num_iters gradient steps with learning rate alpha
  
  % Initialize some useful values
  m = length(y); % number of training examples
  tsize = length(theta); % length of my theta vector
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta_product = zeros(m, tsize);
    for j = 1:tsize
      for i = 1:m
        h_theta = hypothesis(X(:,2)(i), theta) - y(i);
        theta_product(i, j) = h_theta * X(i, j);
      endfor
    endfor
    
    %% simultaneously update
    theta = theta - (alpha * (1/m) * sum(theta_product)');

    % ============================================================
    
    % Save the cost J in every iteration
    # fprintf('[%f, %f]\n', theta(1), theta(2));
    J_history(iter) = computeCost(X, y, theta);
    
  end
  
end
