function [num] = calc_eltsdict(k,n)
%CALC_ELTSDICT Summary of this function goes here
% Written by Monica Jane Emerson, April 2018, MIT License
vect = [1:n];
series = k.^(vect-1);
num = k*sum(series);
end

