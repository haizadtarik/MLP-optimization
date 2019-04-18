function J = nnCostFunction(weights, net, X, y)

input_layer_size = size(X,1);   % Number of inputs
output_layer_size = size(y,1);  % Number of outputs  
hidden_unit = net.layers{1}.dimensions; % Number of hidden units  

Theta1 = reshape(weights(1:hidden_unit*input_layer_size), hidden_unit, input_layer_size);
Theta2 = reshape(weights(1+(hidden_unit*input_layer_size):(hidden_unit*input_layer_size)+(hidden_unit*output_layer_size)), output_layer_size, hidden_unit);      
bias = reshape(weights((hidden_unit*input_layer_size)+(hidden_unit*output_layer_size)+1:end),output_layer_size,1);

net.IW{1,1} = Theta1;
net.LW{2,1} = Theta2;
net.b{2,1} = bias;

yhat = net(X);
%Compute cost function
J = mean(mean((y-yhat).^2));
end
