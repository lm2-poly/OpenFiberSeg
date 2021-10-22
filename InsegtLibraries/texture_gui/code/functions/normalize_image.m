function I = normalize_image(I)
% we work with [0 1] images
if isa(I,'uint8')
    I = double(I)/255;
end
if isa(I,'uint16')
    I = double(I)/65535;
end
end