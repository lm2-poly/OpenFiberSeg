function [patch_size] = calcu_psize(pixPerDiam)
%CALCU_PSIZE Loads a selected region of interest from a volume.
%   Details...
%by Monica Jane Emerson, April 2019, MIT License
if mod(round(pixPerDiam),2)==0
    if (round(pixPerDiam)-pixPerDiam)<0
        patch_size = round(pixPerDiam)+1;
    else
        patch_size = round(pixPerDiam)-1;
    end   
else
    patch_size = round(pixPerDiam);
end
end

