function [a,NN] = forwardprop(NN,X)

m = size(X,1);
NN.layer{1} = X;
NN.net{1} = 0;
for i=1:size(NN.Neurons,2)-1
    a  = [ones(m,1) NN.layer{i}];
    z = a*NN.W{i}';
    a = sigmoid(z);
    NN.layer{i+1} = a;
    NN.net{i+1} = z;
end
% a = a'; %column




% a1 = X
% a1 = [ones(m,1) X];
% z2 =  a1*Theta1'; %5000
% a2 = sigmoid(z2);
% a2 = [ones(m,1) a2];
% 
% z3 = a2*Theta2';
% a3 = sigmoid(z3); % outputs for every training. 5000 x 10 outputs