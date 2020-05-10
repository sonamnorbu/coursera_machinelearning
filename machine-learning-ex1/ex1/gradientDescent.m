function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp=zeros(size(theta),1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    hypothesis = X*theta;       % calculate the hypothesis

     % x1=X(:,1);                  % get the 1st Column from X vector which is just 1s
    %temp1 = theta(1) - alpha/m * sum((hypothesis - y).*x1); % caculate the theta0
    %x2=X(:,2);                  % get the 1st Feature which is column 2 from X vector which is just 1s
    %temp2 = theta(2) - alpha/m * sum((hypothesis - y).*x2);  % caculate the theta1
    %theta(1)=temp1;
    %theta(2)=temp2;

    
  %  for iter2 = 1:size(X,2)
        
   %     x=X(:,iter2);                  % get the 1st Column from X vector which is just 1s
   %     temp(iter2) = theta(iter2) - alpha/m * sum((hypothesis - y).*x); % caculate the theta0
   % end
   % theta=temp;  
    
    %Vectorised

   % theta = theta - (alpha .* (X * theta - y)'*X ./m)';

    h = X * theta;
    errors = h - y;
    delta = X' * errors;
    theta = theta - (alpha / m) * delta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('Loop no: %f Cost computed = %f \n', iter, J_history(iter));
end

end
