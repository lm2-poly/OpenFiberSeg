clear 
close all
addpath functions

% example of re-using externaly made labeling image
im = imread('../data/randen15B.png'); % randen 5 textures
labeling = double(imread('../data/randen_labels_rgb.png'));
dictopt.patch_size = 15;
dictopt.branching_factor = 4;
dictopt.number_layers = 4;
dictopt.number_training_patches = 2000;
dictopt.normalization = false;
dictopt.method = 'euclidean';
image_texture_gui(im,dictopt,5,labeling)

%% an image exported from gui can be re-used 
image_texture_gui(im,dictopt,5) %% <- EXPORT (E) labeling here
image_texture_gui(im,dictopt,5,gui_L) %% <- use labeling here

%% an image saved from gui can be re-used 
image_texture_gui(im,dictopt,5) %% <- SAVE (S) labeling here using some filename
image_texture_gui(im,dictopt,5,imread('filename_labels_indexed.png')) %% <- use labeling here

