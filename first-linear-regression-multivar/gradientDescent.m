function [theta, J_history] = gradientDescent(X, y, theta, alpha, epochs)
  #
  # Para cada interação vamos calcular uma hipotese,
  # depois vamos medir o erro/custo da hipotese e tentar baixar o erro
  # a cada nova interação.
  #
  # Por fim vamos retornar o melhor valor de theta
  #
  
  tsize = length(theta); % length of my theta vector
  m = length(y); % number of examples
  J_history = zeros(epochs, 1);
  hypotheses = zeros(size(X));
  
  for iter = 1:epochs
    for j = 1:tsize
      for i = 1:m
        features = X(i,:)'; # parse matrix to vector of the features
        hypotheses(i, j) = (hypothesis(features, theta) - y(i)) * X(i, j);
      endfor
    endfor
    
    # simultaneously update
    theta = theta - (alpha * (1/m) * sum(hypotheses)');

    # compute the cost for each new theta values
    # fprintf('%.3f\n', computeCost(X, y, theta));
    J_history(iter) = computeCost(X, y, theta);

  endfor

end