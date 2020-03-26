clear; close all;
data=load('simplefit_dataset');
x = data.simplefitInputs;
y = data.simplefitTargets;
% x=scaledata(data.simplefitInputs,0,10); 
% y=scaledata(data.simplefitTargets,0,10);

% x = scaledata

k = [10];            % two hidden layers with 3 and 4 hidden nodes
lambda = 1e-2;
[model, L] = mlpReg(x,y,k);
t = mlpRegPred(model,x);
plot(L);
figure;
hold on
plot(x,y,'.');
plot(x,t);
hold off