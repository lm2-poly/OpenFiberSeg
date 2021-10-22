function [contents_datafolder,path_firstSlice,im] = indicate_dataFolderFunction(path_volumeFolder)
%INDICATE_DATAFOLDER Allows the user to specify the location of their data,
%and displays a slice to ensure that the right data is contained at the 
%specified location.
%   Written by Monica Jane Emerson, March 2019, MIT License

%USER INPUT: Path to CT volume? 
if ~strcmp(path_volumeFolder(end),'/')
    path_volumeFolder = [path_volumeFolder,'/'];
end

%add the path and check the contents of the data folder
addpath(genpath(path_volumeFolder))
contents_datafolder = dir(path_volumeFolder);

%display cross-section, first slice
path_firstSlice = [path_volumeFolder,contents_datafolder(3).name];
im = imread(path_firstSlice);
figure, imagesc(im), axis image, colormap gray, title('First slice of the loaded data-set')
end

