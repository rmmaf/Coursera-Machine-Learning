function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %predicoes para calculo mais simples
    predictions = [];
    for i = 1:m
        x = X(i, :);
        x = x';
        prediction = theta' * x;
        predictions = [predictions, prediction];
    end
    
    delta =[];
    for j = 1:length(X(1, :))
        somatorio = 0.0;
        for i = 1:m
            x = X(i, :);
            x = x';
            somatorio = somatorio + ((predictions(i) - y(i)) * x(j));
        end
        delta = [delta ; ((1/m) * somatorio)];
    end
    
    theta = theta - alpha * delta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
