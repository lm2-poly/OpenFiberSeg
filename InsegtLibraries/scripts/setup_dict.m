%SETUP_DICT Set-ups the default parameters for the dictionary, andasks for the
%data-dependent parameter (patch size)
%   Written by Monica Jane Emerson, March 2019, MIT License

%USER INPUT: select patch size, the closest odd integer to factor*diam/pixel_size 
x = inputdlg('Choose patch size: ','User input',[1,20],{num2str(11)}); %ask for patch size
dictopt.patch_size = str2double(x{1}); %save patch size

%Set the default parameters and initialise dictionary
dictopt.method = 'euclidean';
dictopt.branching_factor = 3; %>=3
dictopt.number_layers = 5; %>=5. The higher the more accurate, but also slower and more computationally heavy
dictopt.number_training_patches = 15000; %at least 10*num_dictAtoms(branching_factor,number_layers)
dictopt.normalization = false; %set to true if the global intensity of the slices varies along the depth of the volume

