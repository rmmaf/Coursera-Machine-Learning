function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
htheta = X * theta;
sub = htheta - y;
quad = sub.^2;
sum1 = sum(quad);
thetaAux = theta;
thetaAux(1) = [];%deleting first matrix row
thetaAux = thetaAux.^2;
sum2 = sum(thetaAux);
J = (sum1/(2 * m)) + ((sum2 * lambda)/(2 * m));

thetaAux = theta;
thetaAux(1) = 0;

for j = 1: size(theta)
    grad(j) = (sum(sub .* X(:, j))/m) + ((thetaAux(j) * lambda)/m);
end







% =========================================================================

grad = grad(:);

end
