clear
close all
addpath functions

% VERSION 1: WITHOUT CORRECTION
% for batch processing dictionary is reused, and manual labeling might need
% correction, so easiest is to build dictionary outside the gui to be able
% to use it later
im = double(imread('../data/slice1.png'))/255;

dictopt.method = 'euclidean';
dictopt.patch_size = 9;
dictopt.branching_factor = 5;
dictopt.number_layers = 4;
dictopt.number_training_patches = 30000;
dictopt.normalization = false;

dictionary = build_dictionary(im,dictopt);

% IMPORTANT:
% once inside gui export dict_labels to workspace (E)
image_texture_gui(im,dictionary,2)
%%
dictionary = update_dictionary(dictionary,gui_dictprob);
%%
figure, imagesc(gui_S), axis image, title('results from gui')

figure
for i=1:5
    I = imread(['../data/slice',num2str(i),'.png']);
    S = process_image(I,dictionary);
    subplot(1,5,i)
    imagesc(S), axis image, title(i), drawnow
end

%%

% VERSION 2: WITH (OR WITHOUT) CORRECTION
% TODO: freezeing (F) and correcting
