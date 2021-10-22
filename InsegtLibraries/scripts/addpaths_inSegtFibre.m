%ADDPATHS_INSEGTFIBRE Makes all the code of Insegt Fibre accesible
%   Written by Monica Jane Emerson, March 2019, MIT License

%USER INPUT: The path of the textureGUI folder in your computer
x = inputdlg('Confirm the default path of the texture_GUI folder: ',...
    'User input',[1,30],{'./texture_gui'});
path_textureGUI = x{1}; 

%Add the functions and Insegt software path 
addpath(genpath(path_textureGUI))
addpath('./scripts')
addpath('./functions')