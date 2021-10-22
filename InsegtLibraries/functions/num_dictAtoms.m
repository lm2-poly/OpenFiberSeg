function [num] = num_dictAtoms(k,n)
%NUM_DICTATOMS Summary of this function goes here
%by Monica Jane Emerson, April 2018, MIT License
vect = [1:n];
series = k.^(vect-1);
num = k*sum(series);
end

