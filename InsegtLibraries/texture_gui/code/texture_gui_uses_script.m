clear 
close all

addpath functions

%% demo: no input
image_texture_gui

%% usage 1: only image (default dictionary options )
im = double(imread('bag.png'))/255; 
image_texture_gui(im)

%% usage 2: image and dictionary options
dictopt.method = 'euclidean';
dictopt.patch_size = 11;
dictopt.branching_factor = 2;
dictopt.number_layers = 5;
dictopt.number_training_patches = 30000;
dictopt.normalization = false;
image_texture_gui(im,dictopt,5)

%% usage 3: image and dictionary
dictionary = build_dictionary(im,dictopt);
image_texture_gui(im,dictionary,2)

%% usage 4: image and mappings
mappings = compute_mappings(im,dictionary);
image_texture_gui(im,mappings,2)

