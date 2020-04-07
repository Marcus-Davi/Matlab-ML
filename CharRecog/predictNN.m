function p = predictNN(NN,X)

[y,~] = forwardprop(NN,X);

[~,p] = max(y,[],2);
end