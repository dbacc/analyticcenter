function [ Ham ] = Ham(A,B,Q, R, S)
  H1 = A - B *( R \ S');
  Ham = [H1, - B * (R\ B');
         -Q + S * (R\S'), -H1'];


end

