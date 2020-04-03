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
features = size(X,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X*theta;

h = sigmoid(z);
J = -(y'*log(h) + (1-y')*log(1-h))/m + lambda*(theta(2:end)'*theta(2:end))/(2*m); %Vectorized. Ignore theta_0

    e = (h - y);
    grad(1,1) = sum(e)/m; %ones
    for feat=2:features
    grad(feat,1) = sum(e.*X(:,feat))/m + lambda*theta(feat)/m;
    end



% =============================================================

end
