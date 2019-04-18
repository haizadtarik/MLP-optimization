clear all; close all; 
  
% =========== Prepare data ===========

%[input,target] = cho_dataset;
[input,target] = abalone_dataset;

X = input;
Y = target;
m = size(X,2);

% split data
test_percentage = 10; % percentage of desired test sets

split = 1-(test_percentage/100);
idx_train = round(split*m);
X_train = X(:,1:idx_train);
Y_train = Y(:,1:idx_train);
m_train = size(X_train,2);

idx_test = round(split*m_train+1);
X_test = X(:,idx_test:end);
Y_test = Y(:,idx_test:end);
m_test = size(X_test,2);

%% =========== Setup MLP (one-layer) ===========

[input_layer_size,~] = size(X);   % Number of inputs
[output_layer_size,~] = size(Y);  % Number of outputs 

hidden_unit = 10;

net = feedforwardnet(hidden_unit);
net = configure(net, X, Y);
initial_nn_weight = [net.IW{1,1}(:) ; net.LW{2,1}(:); net.b{2,1}(:)];


%% ================ Compute Cost function   ================
%  Implement the feedforward part of the neural network to compute the cost only. 
cost = nnCostFunction(initial_nn_weight, net, X, Y);

%Print value
fprintf('Cost :%f\n' , cost);

%% =================== Part 8: Training NN ===================
%  Train the neural network to optimize the weight
epoch = 50;    % iteration
[net, cost] = optimize(net, X_train, Y_train, epoch);

% save('weight_NARX_GDC.mat','Theta1', 'Theta2')
%% ================= Prediction (training) =================
%  Check training accuracy
[err,acc] = evaluate(net, X_train, Y_train);
fprintf('\nTraining Set Accuracy: %f', acc);
fprintf('\nTraining Set Error: %f', err);
fprintf('\n');

%% ================ Prediction (test) ======================
%  Check test accuracy
[err,acc] = evaluate(net, X_test, Y_test);
fprintf('\nTest Accuracy: %f', acc);
fprintf('\nTest Error: %f', err);
fprintf('\n');


%% ================ Plot ======================
ytest = net(X_test);
for i = 1:size(Y_test,1)
    figure(i)
    plot(Y_test(i,:))
    hold on;
    plot(ytest(i,:))
    legend('Target', 'Predicted')
end


function [error, accuracy] = evaluate(net, x, y)
    yhat = net(x);
    [accuracy,~,~] = regression(y,yhat);
    error = mean((y-yhat).^2);
end
