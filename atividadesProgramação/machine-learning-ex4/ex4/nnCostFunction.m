function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
pred = predict(Theta1, Theta2, X);
K = num_labels;
newPred = zeros(m, K);
newY = zeros(m, K); %matrix for vectorization

for i = 1:m
    newY(i, y(i)) = 1;
    newPred(i, pred(i)) = 1;
end
sub = zeros(m, K);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

for k = 1:K
    first = (-newY(:, k)) .* log(h2(:, k));
    second = (1 - newY(:, k)) .* log(1 - h2(:, k));
    sub(:, k) = first - second;
end
soma1 = sum(sum(sub));
thetaAux1 = Theta1 .^ 2;
thetaAux1(:,1) = [];
thetaAux2 = Theta2 .^ 2;
thetaAux2(:,1) = [];
soma2 = sum(sum(thetaAux2)) + sum(sum(thetaAux1));
J = (soma1/m) + ((soma2 * lambda)/(2*m));

X1 = [ones(m, 1) X];
for t = 1:m
    a1 = X1(t, :);
    z2 = Theta1 * a1.';
    a2 = sigmoid(z2);
    a2 = [1 ; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    miniDelta3  = a3 - newY(t, :).';
    miniDelta2 = ((Theta2.')*miniDelta3) .* sigmoidGradient([1; z2]);
    
    miniDelta2 = miniDelta2(2:end);
    if(t == 1)
        delta1 = zeros(size(miniDelta2 * a1));
        delta2 = zeros(size(miniDelta3 * a2.'));
    end
    delta1 = delta1 + miniDelta2 * a1;
    delta2 = delta2 + miniDelta3 * a2.';
    
end

Theta1_grad = delta1/m;
Theta2_grad = delta2/m;
 











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
