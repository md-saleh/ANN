function numgrad = fdGradient(J, theta)

%computing gradient using finite difference     

numgrad = zeros(size(theta));
eta = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    eta(p) = e;
    loss1 = J(theta - eta);
    loss2 = J(theta + eta);
    numgrad(p) = (loss2 - loss1) / (2*e);
    eta(p) = 0;
end

end
