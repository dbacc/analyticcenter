function [ H ] = H(A,B,Q, R, S, X)

[n,m] = size(B);
H = [ Q, S ;
    S', R]...
    - [A'* X  + X*A, X *B;
    B'*X, zeros(m,m)];
    

end

