clear; close all; clc;

input_layer_size  = 2;
hidden_layer_size = 25;
output_layer_size = 1;

x = [-2:0.001:2];
y = [-2:0.001:2];

X = [x' y'];

out = exp(-x.^2 - y.^2)';

subplot(1, 2, 1); plot(out);

% i use epsilon to keep weights small and learning efficient
epsilon = sqrt(6)/(sqrt( 25 + 3));
Theta1 = rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon - epsilon;  % 25 x 3
epsilon = sqrt(6)/(sqrt( 1 + 26));
Theta2 = rand(output_layer_size, hidden_layer_size + 1)* 2 * epsilon - epsilon; % 1  x 26

nn_weights = [Theta1(:) ; Theta2(:)];

%feed forward and back propagation
lambda = 0; %weight regulization

[J grad] = nnCostFun(nn_weights, input_layer_size, hidden_layer_size, output_layer_size, X, out, lambda);

[nn_params, cost] = gradientDescent(X, out, nn_weights, grad, 0.9, 500);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
       
% visualize output
m = size(X, 1);
       
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
H = sigmoid(Z3);

subplot(1, 2, 2); plot(H);

figure; plot(cost);

% Testing
xnew = [0:0.1:5];
ynew = [0:0.1:5];
Xnew = [xnew' ynew'];
outnew = exp(-xnew.^2 - ynew.^2)';

m = size(Xnew, 1);

A1 = [ones(m, 1) Xnew];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
H = sigmoid(Z3);

figure;subplot(1,2,1);plot(outnew);subplot(1,2,2);plot(H);