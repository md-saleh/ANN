clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

load('data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

epsilon = sqrt(6)/(sqrt( 25 + 401));
Theta1 = rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon - epsilon;  % 25 x 3
epsilon = sqrt(6)/(sqrt( 1 + 26));
Theta2 = rand(num_labels, hidden_layer_size + 1)* 2 * epsilon - epsilon; % 1  x 26

load('weights.mat');

nn_weights = [Theta1(:) ; Theta2(:)];

lambda = 1;

[J, grad]= nnCostFun(nn_weights, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);


%options = optimset('MaxIter', 50);

%costFunction = @(p) nnCostFun(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

%[nn_params, cost] = fmincg(costFunction, nn_weights, options);

%Theta1 = reshape(nn_weights(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

%Theta2 = reshape(nn_weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
                                  
[nn_weights, cost] = gradientDescent(X, y, nn_weights, grad, 0.1, 50);

Theta1 = reshape(nn_weights(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

%save('weights.mat', 'nn_weights');

pred = predict(Theta1, Theta2, X);

preds = pred(sel, :);
S = preds(1:10, :)';
for i = 2:10
  S = [S; preds(((i-1)*10)+1:i*10, :)'];
end

S

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


