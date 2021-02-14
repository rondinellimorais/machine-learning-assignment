function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta.
  %               You should set J to the cost.
  %               Compute the partial derivatives and set grad to the partial
  %               derivatives of the cost w.r.t. each parameter in theta
  
  # OBS:
  # No exercicío 3 lrCostFunction fizemos uma lógica diferente
  # da uma lá e vê qual é a melhor.
  
  # calcule cost
  hypot = sigmoid(X * theta);
  result = (-y .* log(hypot)) - ((1 - y) .* log(1 - hypot));
  
  cost_regularize_term = 0;
  for j = 1:length(theta)
    if j >= 2
      cost_regularize_term += (lambda / (2*m)) * sum(theta(j) .^ 2);
    endif
  endfor
  
  J = (1/m) * sum(result + cost_regularize_term);

  # calcule gradient
  grad = (1 / m) * sum((hypot - y) .* X);
  for j = 1:length(grad)
    # we do not want to penalize theta(1)
    if j >= 2
      grad(j) = grad(j) + ((lambda/m) * theta(j));
    endif
  endfor
 
end
