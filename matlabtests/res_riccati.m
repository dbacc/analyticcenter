function [ res ] = res_riccati( A,B,Q, R, S, X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
F = R\(S' - B' *X);
res = Q - A'* X  - X*A - F'*R*F;
end

