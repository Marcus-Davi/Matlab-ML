clear;close all;clc
%% Load Data

m = 2000; %all samples
m_test = 150;
[X_, y_] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',m,0);
[X_t, y_test] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',m_test,0);
y_(y_==0) = 10;

y = y_(1:m);

for i=1:m
    unroll = X_(:,:,i);
    unroll = unroll(:);
   X_train(i,:) = unroll'; 
end


for i=1:m_test
    unroll = X_t(:,:,i);
    unroll = unroll(:);
   X_test(i,:) = unroll'; 
end

num_labels = 10;
N_I = size(X_train,2);
N_O = num_labels; %labels

% imag = input(:,:,2);
% imshow(imag)

%% NN Layers
Hidden_Layers = 1;
N_2 = 30; %hidden neurons

%% Matrices
epslon = 0.12;
Theta1 = rand(N_2,N_I+1)*epslon*2 - epslon;
Theta2 = rand(N_O,N_2+1)*epslon*2 - epslon;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

nn_params = [Theta1(:); Theta2(:)];
% %% Training (Manual, Gradient Descent)
% Max_Iterations = 50;
% lambda = 1;
% l_rate = linspace(0.8,1.5,Max_Iterations);
% J_history = zeros(Max_Iterations,1);
% 
% batch_size = 100;
% 
% for it=1:Max_Iterations
%     
%     [J, Grad] = nn_cost(nn_params,N_I,N_2,num_labels,X_train,y,lambda);
%     J_history(it) = J;
%     nn_params = nn_params - l_rate(it)*Grad;
%     fprintf('Iteration %d - Cost %f \n',it,J);
% end
% 
% Theta1 = reshape(nn_params(1:N_2 * (N_I + 1)), ...
%                  N_2, (N_I + 1));
% 
% Theta2 = reshape(nn_params((1 + (N_2 * (N_I + 1))):end), ...
%                  num_labels, (N_2 + 1));
%              
%% Training (Optimal)

options = optimset('MaxIter', 100);
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nn_cost(p, ...
                                   N_I, ...
                                   N_2, ...
                                   num_labels, X_train, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);


% Predict

Theta1 = reshape(nn_params(1:N_2 * (N_I + 1)), ...
                 N_2, (N_I + 1));

Theta2 = reshape(nn_params((1 + (N_2 * (N_I + 1))):end), ...
                 num_labels, (N_2 + 1));

             
%% Test Trained cases
p = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);

%% Test Untrained cases
 % Prediction
p_test = predict(Theta1,Theta2,X_test);
    
fprintf('\n Test Set Accuracy: %f\n', mean(double(p_test == y_test)) * 100);

%% Test Loop
fprintf('Test loop ...');
while true
    test_index = randi(m_test);
    img = X_t(:,:,test_index);
    imshow(img);
    x_test = img(:);
    
    p_test = predict(Theta1, Theta2, X_test(test_index,:)); 
    fprintf('Guessed number = %d\n',p_test);
      
    pause;
end

