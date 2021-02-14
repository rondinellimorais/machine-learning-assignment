# Aqui vamos escrever nosso primeiro algoritmo de regressão linear
# com multiplas variáveis.
#
# O desafio aqui é utilizar a predição do preço de uma casa
# utilizando as técnicas que aprendemos até aqui
#

# Passos
# ====================================
# 1. Carregar o dataset
# 2. Mostrar os dados
# 3. Normalizar os dados
# 4. Utilizar gradient descent para encontrar theta
#   4.1. Calcular o custo a cada interação.
#        OBS: O custo é a distância da hipotese ao dado original o custo
#        deve diminuir a cada interação.
# 5. Ajustar o modelo aos dados (hypothesis)
# 6. Realizar uma predição
#


# load dataset
# ========================================
dataset = csvread('prices.csv');
m = length(dataset); # number of examples
dataset = dataset(2:m,:); # skip header

# show data
# ========================================
fprintf('\nload dataset\n');
display(dataset(1:5,:));

# feature normalize
# ========================================
dataset_norm = featureNormalize(dataset(:,1:2));
X = [ones(length(dataset_norm), 1) dataset_norm];
y = dataset(:,3);

# show features normalized
# ========================================
fprintf('\nfeatures normalized\n');
display(X(1:5,:));

# calcule theta parameter
# ========================================
fprintf('\nRunning gradient descent ...\n');

theta = zeros(size(X, 2), 1);
alpha = 0.01;
epochs = 500;
[theta, J_history] = gradientDescent(X, y, theta, alpha, epochs);

# plot cost function values
figure;
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 1);
xlabel('Number of iterations');
ylabel('Cost J');

# Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

# plot linear regression model
plotModel(X, y, theta);

# Predict values
predict_scaling = theta' * [1.0000;-0.78102;-0.22368];
fprintf('The price of the house is %.2f\n', predict_scaling);

# Parece que conseguimos criar nosso primeiro algoritmo de regressão linear
# com multiplas variáveis
# Parabéns!!!

# Ele tem alguns erros em relação a precisão mas isso é algo
# que vamos ainda ver durante o curso.
#
# fim