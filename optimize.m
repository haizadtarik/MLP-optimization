function [net, cost] = optimize(net, X, y, iteration)
% Optimize the neural network weight

input_layer_size = size(X,1);   % Number of inputs
output_layer_size = size(y,1);  % Number of outputs  
hidden_unit = net.layers{1}.dimensions; % Number of hidden units  

costFunction = @(p) nnCostFunction(p, net, X, y);
nvars = size([net.IW{1,1}(:) ; net.LW{2,1}(:); net.b{2,1}(:)],1);
lb = ones(1,nvars)*(-4);
ub = ones(1,nvars)*(4.2);
SearchAgents = nvars*10;

% Enable(uncomment) desired optimization method

% =================== levenberg-marquardt ==================
% options =  optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'MaxIterations', iteration, 'display','iter', 'MaxFunctionEvaluations',60000); 
% x0 = [net.IW{1,1}(:) ; net.LW{2,1}(:); net.b{2,1}(:)];
% [weights_optimize,cost] = lsqnonlin(costFunction,x0, [], [], options);

% =================== GA ==================
% ga_opts = gaoptimset('Generations', iteration, 'display','iter', 'PopulationSize', SearchAgents, 'StallGenLimit',5);
% [weights_optimize, cost] = ga(costFunction ,nvars,[],[],[],[],lb,ub,[],ga_opts);

% =================== PSO ==================
options =   optimoptions('particleswarm', 'MaxIterations',iteration, 'display','iter','SwarmSize',SearchAgents); 
[weights_optimize, cost] = particleswarm(costFunction ,nvars,lb,ub,options);

% =================== SA ==================
% alpha = 0.8;
% [weights_optimize,cost]=sa_con(costFunction,nvars,lb,ub,iteration,alpha);

% =================== FPA ==================
% probability =0.2;
% [weights_optimize, cost]=FPA(costFunction ,nvars,lb,ub,SearchAgents,probability,iteration);

% =================== BAT ==================
% Loudness = 0.7;
% pulse_rate = 0.7;
% [weights_optimize, cost]=bat_algorithm(costFunction,nvars,lb,ub,SearchAgents,iteration,Loudness,pulse_rate);

% =================== FFA ==================
%[weights_optimize,cost]=ffa_con(costFunction,nvars,lb,ub,SearchAgents,iteration);

% =================== Cuckoo ==================
% [weights_optimize,cost] = cuckoo_search(costFunction,nvars,lb,ub,SearchAgents,iteration);


% =================== SSA ==================
% addpath 'SSA';
% [cost,weights_optimize,~]=SSA(SearchAgents,iteration,lb,ub,nvars,costFunction);

% =================== WOA ==================
% addpath 'WOA';
% [cost,weights_optimize,~]=WOA(SearchAgents,iteration,lb,ub,nvars,costFunction);

% =================== ALO ==================
% addpath 'ALO';
% [cost,weights_optimize,~]=ALO(SearchAgents,iteration,lb,ub,nvars,costFunction);


% ----------------------- Obtain Theta1, Theta2 and bias back from weights_optimize -------------------------------------
Theta1 = reshape(weights_optimize(1:hidden_unit*input_layer_size), hidden_unit, input_layer_size);
Theta2 = reshape(weights_optimize(1+(hidden_unit*input_layer_size):(hidden_unit*input_layer_size)+(hidden_unit*output_layer_size)), output_layer_size, hidden_unit);      
bias = reshape(weights_optimize((hidden_unit*input_layer_size)+(hidden_unit*output_layer_size)+1:end),output_layer_size,1);

net.IW{1,1} = Theta1;
net.LW{2,1} = Theta2;
net.b{2,1} = bias;
end