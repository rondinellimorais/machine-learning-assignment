function J = computeCost(X, y, theta)
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  J = 0;
  
  vector = zeros(1, m);
  for i = 1:m
    features = X(i,:)';
    vector(i) = ((theta' * features) - y(i)) .^ 2;
  endfor
  
  J = 1 / (2 * m) * sum(vector);

end
