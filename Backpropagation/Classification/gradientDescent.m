function [theta, hist] = gradientDescent(X, y, theta, delta, alpha, num_iters)
  
hist = zeros(num_iters, 1);

for iter = 1:num_iters

theta = theta - alpha .* delta;
   
[hist(iter) delta] = nnCostFun(theta, 400, 25, 10, X, y, 1);

end

end