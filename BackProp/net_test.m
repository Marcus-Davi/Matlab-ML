clear;close all;clc

load('Net_Boa'); %nome da net = 'net'

x = -2*pi:0.1:2*pi;

y = netrun(net,x);
plot(x,y)
hold on
plot(x,x.^2 + 10)
legend('rede neural do davi','oq ela deveria imitar')
grid on