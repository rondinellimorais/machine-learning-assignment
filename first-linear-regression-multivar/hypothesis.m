function J = hypothesis(features, theta)
  # Calculo da hipotese
  # h(xi) = θ0 + θ1 * X(1) + θ2 * X(2) + θn * X(n)
  #
  # Em resumo: Cada valor de theta multiplicado por cada coluna (feature)
  # para cada linha e no final soma todas as colunas da mesma linha
  #
  J = theta' * features;
end