function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
  %   taking num_iters gradient steps with learning rate alpha
  
  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  tsize = length(theta); % length of my theta vector
  
  for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    theta_product = zeros(m, tsize);
    for j = 1:tsize
      for i = 1:m
        h_theta = (theta' * X(i,:)') - y(i);
        theta_product(i, j) = h_theta * X(i, j);
      endfor
    endfor
    
    %% simultaneously update
    theta = theta - (alpha * (1/m) * sum(theta_product)');
    
    % ============================================================
    
    % Save the cost J in every iteration
    display(theta);
    J_history(iter) = computeCostMulti(X, y, theta);
    
  end
  
end
