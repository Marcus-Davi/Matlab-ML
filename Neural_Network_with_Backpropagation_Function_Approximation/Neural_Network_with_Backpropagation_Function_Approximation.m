%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all; tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NN Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nIter = 1e4;        alpha = 0.01;       nInputs = 1;        nOutputs = 1;
f1='logsig';        f2 = 'purelin';     f3  = 'tansig';     f4 = 'hardlim';
% Function to be approximated ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x = linspace(-2,2,1e2);                 y = 1+sin((pi/4)*x);     

% load('simplefit.mat')
% x = simplefitInputs;
% y = simplefitTargets;
% x = scaledata(simplefitInputs,0,1);
% y = scaledata(simplefitTargets,0,1);
% y = 1+sin((pi/4)*x);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%%%%%%%%%%% Neural Network with Backpropagation: Configuration %%%%%%%%%%%
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
display('=============== Neural Network with Backpropagation: Configuration ===============');
nLayers = input(strcat('(1). Number of network layers is:'));
for i=1:1:nLayers
    
    if mod(i,10)==1 
        strng='st';
    elseif mod(i,10)==2
        strng='nd';
    elseif mod(i,10)==3
        strng='rd';
    else
        strng='th';
    end
    
    if (i~=nLayers)
        layer(i).nPerc  = input(strcat('(2).[',int2str(i),'/',...
            int2str(nLayers),'] nPerc. in the [',int2str(i),strng,'] layer is:'));
    else
        layer(i).nPerc  = nOutputs;
    end
    
    if i==1
        layer(i).bias   = randn(layer(i).nPerc,1);
        layer(i).weight = randn(layer(i).nPerc,nInputs);
    elseif i==nLayers
        layer(i).bias   = randn(nOutputs,1);
        layer(i).weight = randn(nOutputs,layer(i-1).nPerc);
    else
        layer(i).bias   = randn(layer(i).nPerc,1);
        layer(i).weight = randn(layer(i).nPerc,layer(i-1).nPerc);
    end
    layer(i).func       =  f1;
end
layer(nLayers).func    =  f2;
% Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clc; display('(3). Now Training...'); pause(2);
tstart=tic;
for i=1:1:nIter
    display('==============================================================');
    display(strcat('Training [%',num2str(i/nIter*100),'] Completed'));
    display('==============================================================');
    % Training Input Selection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    indx = randi(length(y),1,nInputs);
    a0   = x(indx);
%     trgt = 1+sin(pi/4*a0);
    trgt = y(indx);
        % Feedforward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for j=1:1:nLayers
        if (j==1)
            layer(j).out = feval(layer(j).func,layer(j).weight*a0+layer(j).bias);
        elseif(j==nLayers)
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        else
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        end
    end
    err(:,i) = trgt-layer(nLayers).out;
    % Backpropagation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (strcmp(layer(nLayers).func,'logsig')||strcmp(layer(nLayers).func,'tansig'))
        layer(nLayers).dfunc=feval(strcat('d',layer(nLayers).func),1,layer(nLayers).out);
    else
        layer(nLayers).dfunc=1;
    end
    
    layer(nLayers).sensitivity=-2*diag(layer(nLayers).dfunc)*err(:,i);
    
    for k=nLayers-1:-1:1
        if (strcmp(layer(k).func,'logsig')||strcmp(layer(k).func,'tansig'))
            layer(k).dfunc=feval(strcat('d',layer(k).func),1,layer(k).out);
        else
            layer(k).dfunc=feval(strcat('d',layer(k).func),layer(k).out);
        end
        layer(k).sensitivity = diag(layer(k).dfunc)*layer(k+1).weight'*layer(k+1).sensitivity;
    end
    % Updating Weights and Bias ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for z=1:1:nLayers
        layer(z).bias   = layer(z).bias  -alpha.*layer(z).sensitivity;
        if (z~=1)
            layer(z).weight = layer(z).weight-alpha.*layer(z).sensitivity*layer(z-1).out';
        else
            layer(z).weight = layer(z).weight-alpha.*layer(z).sensitivity*a0';
        end
    end
   clc; 
end

clc; display('Training: Done!!'); 
display(strcat('Execution Time: [',num2str(toc(tstart)),'] seconds.')); pause(2);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Experimental Calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
apprx=zeros(1,length(x));
for kk=1:1:length(x)
    a0=x(kk);
    for j=1:1:nLayers
        if (j==1)
            layer(j).out = feval(layer(j).func,layer(j).weight*a0+layer(j).bias);
        elseif(j==nLayers)
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        else
            layer(j).out = feval(layer(j).func,layer(j).weight*layer(j-1).out+layer(j).bias);
        end
    end
    apprx(kk)=layer(nLayers).out;
end
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Plotting the Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
figure; semilogx(1:i,(err(1:i))); grid on;
xlabel('Iteration'); ylabel('Error (Iteration)');

figure; plot(x,y,x,apprx,'o');
xlabel('x');ylabel('f(x)');
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%