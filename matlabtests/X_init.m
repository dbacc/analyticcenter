function [ X, alpha ] = X_init( A,B,Q,R,S, Xm, Xp )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
alpha = 0.001;
[m,n] = size(A);
for i = 1:100
   alpha = 2 * alpha; 
   X = Xm + alpha* eye(n);
   if min(eig(H(A,B,Q,R,S,X))) > 1.1
       break
   end
end


end

