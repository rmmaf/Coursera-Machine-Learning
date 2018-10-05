function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = [];
for i = 1:m
    x = X(i, :);
    x = x';
    prediction = sigmoid(theta' * x);
    predictions = [predictions, prediction];
end

somatorio = 0.0;
for i = 1:m
    somatorio = somatorio + ((-y(i)*log(predictions(i))) - ((1 -y(i)) * log(1 - predictions(i))));
end

somatorio = somatorio / m;

for j = 1:n
    if j ~= 1
    somatorio = somatorio + (lambda/(2*m)) * theta(j)^2;
    end
end
J = somatorio;

for j = 1:length(X(1, :))
    somatorio = 0.0;
    for i = 1:m
        if(j == 1)
            somatorio = somatorio + (predictions(i) - y(i))*X(i, j);
        else
            somatorio = somatorio + (predictions(i) - y(i))*X(i, j) + (lambda/(m))*theta(j);
        end
    end
    theta(j) = somatorio/m;
end
grad = theta;
% =============================================================

end
