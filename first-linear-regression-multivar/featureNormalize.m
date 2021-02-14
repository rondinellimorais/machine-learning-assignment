function dataset_norm = featureNormalize(dataset)
  # How to feature normalize works?
  # ===============================
  # column i minus mean of the column i divided by
  # standard deviation of column i
  #
  # i.e
  # dataset iquals to:
  #   5   6   7
  #   8   9   10
  #   11  12  13
  #
  # column 1 equals to:
  #   5
  #   8
  #   11
  #
  # for each row we have:
  #   5 - mean(5,8,11) / std(5,8,11)
  #   8 - mean(5,8,11) / std(5,8,11)
  #   11 - mean(5,8,11) / std(5,8,11)
  #

  dataset_norm = zeros(size(dataset));
  n_features = size(dataset, 2);

  for i = 1:n_features
    std_matrix = std(dataset(:,i));
    dataset_norm(:,i) = (dataset(:,i) - mean(dataset(:,i))) / std_matrix;
  endfor
  
end