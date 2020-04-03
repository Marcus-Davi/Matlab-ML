function out = netrun(net,input)
n_layer = length(net.layer);

% Foward
layer_input = input;

for l = 1:n_layer
    
   z = net.layer(l).W*layer_input + net.layer(l).Bias;
   if(strcmp(net.layer(l).act,'puresim'))
   out = z;    
   else
   out = feval(net.layer(l).act,z);
   end
   
   layer_input = out;
    
end



end