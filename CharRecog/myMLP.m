clear;close all;clc
%% Load Data

m = 2000; %all samples
m_test = 100;
[X_, y_] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',m,0);
[X_t, y_test] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',m_test,0);
y_(y_==0) = 10;

y = y_(1:m);

for i=1:m
    unroll = X_(:,:,i);
    unroll = unroll(:);
   X_train(i,:) = unroll';   
   y_train(i,y(i)) = 1;
   
end

XX = X_train';
YY = y_train'
return

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

%% NN Struct Class
NN.HiddenLayers = 2;
NN.Neurons = [N_I zeros(1,NN.HiddenLayers) N_O]; 

for i=1:NN.HiddenLayers
    fprintf('Type number of neurons at hidden layers #%d: ',i)
    NN.Neurons(i+1) = input('');
end

% Bypass
% NN.Neurons(2) = 30;
layers = size(NN.Neurons,2);
for layer=1:layers
    NN.layer{layer} = zeros(NN.Neurons(layer),1);
end
%% Weigth Matrices
epslon = 0.12;
rng(1)
for i=1:size(NN.Neurons,2)-1
   NN.W{i} = rand(NN.Neurons(i+1),NN.Neurons(i)+1)*2*epslon-epslon; % +1 para incluir bias
   NN.G{i} = zeros(size(NN.W{i})); %Gradients
    %Incluir uNROLL
end
% NN
% return

%% Training
Iterations = 200;
J = 0;
layers = size(NN.Neurons,2);
l_rate = linspace(1.0,1.7,Iterations);
lambda = 1;



J_history = zeros(Iterations,1);
for it = 1:Iterations
    
    for layer=1:layers-1
       NN.G{layer} = NN.G{layer}*0; 
    end
    
    for i=1:m

        % forward prop
        y_ = y_train(i,:);
        [y_nn,NN] = forwardprop(NN,X_train(i,:));
        y_(y(i)) = 1;% converts y(5000 x 1) to 5000 x 10 outputs
        %cost
        J = J + -y_*log(y_nn')-(1-y_)*log(1-y_nn');
%           J = J + -y_'*log(y_nn')-(1-y_')*log(1-y_nn');
        delta = y_nn' - y_';
        % backprop
       for layer=(layers-1):-1:1
           NN.G{layer} = NN.G{layer} + delta*[1 NN.layer{layer}];
           delta = NN.W{layer}'*delta;
           delta = delta(2:end).*sigmoidGradient(NN.net{layer})';
       end


    end
    
    J = J/m;
    S = 0;
     
    for layer=1:layers-1
    NN.G{layer} = NN.G{layer}/m;
    NN.G{layer}(:,2:end) = NN.G{layer}(:,2:end) + lambda*NN.W{layer}(:,2:end)/m;
    T = NN.W{layer}(:,2:end).*NN.W{layer}(:,2:end);
    S = S + sum(T(:));
    end
    Reg = lambda*S/(2*m); %Regularization
    
    J = J + Reg;
    J_history(it) = J;
    
    % Grad Update
    for layer=1:layers-1
    NN.W{layer}  = NN.W{layer} - l_rate(it)*NN.G{layer};
    end
    fprintf('Iteration %d - Cost %f \n',it,J);
end

plot(J_history)

% Predict             
%% Test Trained cases
p = predictNN(NN, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);

%% Test Untrained cases
 % Prediction
p_test = predictNN(NN,X_test);
    
fprintf('\n Test Set Accuracy: %f\n', mean(double(p_test == y_test)) * 100);

%% Test Loop
fprintf('Test loop ...');
while true
    test_index = randi(m_test);
    img = X_t(:,:,test_index);
    imshow(img);
    x_test = img(:);
    
    p_test = predictNN(NN, X_test(test_index,:)); 
    fprintf('Guessed number = %d\n',p_test);
      
    pause;
end

