function plotModel(X, y, theta)
  
  X_size = X(:,2);
  X_nbedrooms = X(:,3);
  m = length(y);
  
  figure; % open a new figure window
  
  # first plot
  subplot(1, 2, 1);
  plot(X_size, y, 'rx', 'MarkerSize', 5);
  xlabel('size of the house');
  ylabel('price');
  title('Size of the house x Price');
  
  hold on;
  plot(X_size, [ones(m, 1) X_size]*[theta(1); theta(2)], '-');
  legend('Training data', 'Linear regression');
  hold off;
  
  # second plot
  subplot(1, 2, 2);
  plot(X_nbedrooms, y, 'bx', 'MarkerSize', 5);
  xlabel('number of bedrooms');
  ylabel('price');
  title('Number of bedrooms x Price');
  
  hold on;
  plot(X_nbedrooms, [ones(m, 1) X_nbedrooms]*[theta(1); theta(3)], '-');
  legend('Training data', 'Linear regression');
  hold off;
  
end
