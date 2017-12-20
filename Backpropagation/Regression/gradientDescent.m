function [theta, hist] = gradientDescent(X, y, theta, delta, alpha, num_iters)
  
hist = zeros(num_iters, 1);

for iter = 1:num_iters

theta = theta - alpha .* delta;
   
[hist(iter) delta] = nnCostFun(theta, 2, 25, 1, X, y, 0);

end

end