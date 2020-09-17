function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
##for j=1:size(z)(1)
##  for k=1:size(z)(2)
##    g(j, k) = 1.0 / (1.0 + e**(-z(j, k)));
##  endfor
##endfor

g = 1 ./ (1 + e.**(-z));



% =============================================================

end
