function [J grad] = nnCostFun(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%forward pass

A1 = [ones(m, 1) X]; %adding bias to input
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
H = A3 = sigmoid(Z3);

sqrErr = (H - y).^2;

J = 1 / (2*m) * sum(sqrErr); %mean squared error as our cost function

%back propagation

Sigma3 = A3 - y;
Sigma2 = (Sigma3 * Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end); %removed sigma 0
Delta_1 = Sigma2' * A1;
Delta_2 = Sigma3' * A2;

Theta1_grad = Delta_1 ./ m + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2 ./ m + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end