clear;close all;clc
%% Load Data
load('simplefit.mat')
% inputs = simplefitInputs;
% targets = simplefitTargets;
% inputs = scaledata(simplefitInputs,0,10);
% targets = scaledata(simplefitTargets,0,10);
inputs = linspace(-5,5,100);                 
targets = inputs.^3 + 2*inputs.^2 - inputs.^4 +2*inputs + 2;  
targets = scaledata(targets,0,1);
% 

test_samples = length(inputs);
n_inputs = 1;
n_outputs = 1;

mingrad = 1e-6;

% return
%% MLP Characteristicz
n1 = 15; % Neurons per Hidden Layer
% n2 = 16;
%% Weights

l_rate = 0.1;

%% Hidden Layers (automatizar/melhorar dps PFRV)
W_input = randn(n1,n_inputs);
B_input = randn(n1,1);
W = randn(n_outputs,n1); % 

B = randn(n_outputs,1);    

% % Gradients
% gW = zeros(n_outputs,n1);
% gB = zeros(n_outputs,1); 
% 
% gW_input = zeros(n1,n_inputs);
% gB_input = zeros(n1,1);




%% Loop
MAXIterations = 100000;

gW = zeros(n_outputs,n1);
gB = zeros(n_outputs,1); 

gW_input = zeros(n1,n_inputs);
gB_input = zeros(n1,1);
L = [];
for it=1:MAXIterations

% it
 i = randi(length(targets),1,n_inputs);
 
current_input = inputs(i);
current_target = targets(i);

% Forward Propagation 
z_hidden = W_input*current_input + B_input;
a_hidden = activation(z_hidden); %layer 1
% Hidden Layers

z_out = W*a_hidden + B;
a_out = (z_out); %PURESIM

error = current_target - a_out;


% Backward Propagation

%Output layer
for j=1:n_outputs
   for k=1:n1
%       gW(j,k) = gW(j,k) - error(j)*activation_deriv(z_out(j))*a_hidden(k);
%         gW(j,k) =  -error(j)*(a_out*(1-a_out))*a_hidden(k); %LOGISG
        gW(j,k) =  -error(j)*a_hidden(k); % PURESIM
%         gW(j,k) =  gW(j,k) -error(j)*(a_out*(1-a_out))*a_hidden(k);
   end
%     gB(j) =  gB(j) - error(j)*activation_deriv(z_out(j)); %output biases
    gB(j) = - error(j); %output biases
%     gB(j) = gB(j) - error(j)*(a_out*(1-a_out)); %output biases
end

% Hidden layer
for j=1:n1 %linha
    for k=1:n_inputs %coluna
        sum = 0;
        for n=1:n_outputs
%            sum = sum - error(n)*activation_deriv(z_out(n))*W(n,j);
           sum = sum - error(n)*W(n,j);
        end
        
        gW_input(j,k) =   sum*activation_deriv(z_hidden(j))*current_input;
%         gW_input(j,k) = gW_input(j,k) - sum*activation_deriv(z_hidden(j))*current_input;
    end
    gB_input(j) =   sum*activation_deriv(z_hidden(j)); %biases
%     gB_input(j) = gB_input(j) - sum*activation_deriv(z_hidden(j)); %biases
end

 %Updates
W = W - l_rate*gW;
B = B - l_rate*gB;
W_input = W_input - l_rate*gW_input;
B_input = B_input - l_rate*gB_input;

allgrad = gW(:) + gB(:) + gW_input(:) + gB_input(:);
L = [L;norm(allgrad)];
if(L(it) < mingrad)
    break
end


% newout = W*activation(W_input*current_input + B_input) + B
% error
% newerror = current_target - newout

end





%% Test & Plots
OUT = zeros(test_samples,1);
for i=1:test_samples
    
% Forward Propagation 
z_hidden = W_input*inputs(i) + B_input;
a_hidden = activation(z_hidden); %layer 1
% Hidden Layers

z_out = W*a_hidden + B;
a_out = (z_out);

correct_out = targets(i);
error = correct_out - a_out;
OUT(i) = a_out;    

net.layer(1).W = W_input;
net.layer(1).Bias = B_input;
net.layer(1).act = 'tanh';

net.layer(2).W = W;
net.layer(2).Bias = B;
net.layer(2).act = 'puresim';    
end




%% Plots

plot(inputs,targets)
hold on
plot(inputs,OUT)
legend('targets','net')
figure
plot(L)
disp('Norma Grad')
disp(L(end))

resp  = input('Salva rede? [s] para sim: ','s');
if(strcmp(resp,'s'))
save('Net_Boa','net')
disp('Rede Salva')
end

function y = activation_deriv(x)
%     y = activation(x)*(1-activation(x));
% y = logsig(x).*(1-logsig(x));
y = 1-tanh(x).^2;
end

function y = activation(x)
% y = 1./(1+exp(-x));
% y = logsig(x);
y = tanh(x);
end
