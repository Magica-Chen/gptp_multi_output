function [L_B,B] = vec2mat_diag(w,d)
% vec2mat_diag - Filled the low triangle matrix by a vector.
%
%%
L_B = zeros(d,d);
for j = 1:d
    A = diag(w(1:d - j +1),-j+1);
    L_B = L_B + A;
    w(1:d+1-j) = [];
end
B = L_B*L_B';
end

