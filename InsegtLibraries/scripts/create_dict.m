%CREATE_DICT Updates the dictionary of intensities built by SETUP_DICT with
%a training data-set that can be loaded or created from scratch.
%   Written by Monica Jane Emerson, March 2019, MIT License

%USER INPUT: Option to load a labelling that was saved

%select a small RoI for training, containing 100 fibres approx
prompt = ['Choose a slice for training: [1,', num2str(size(V_hist,3)),']:'];
x = inputdlg(prompt,...
'User input',[1,30],{num2str(round(size(V_hist,3)/2))});  %select slice

[~,im_train] = selectFoV_VolumetricTiff(V_hist(:,:,str2double(x{1})) );

dictionary = build_dictionary(im_train,dictopt); %create the dictionary of intensities
image_texture_gui(im_train,dictionary,2) %learn the dictionary of probabilities by annotating in the GUI 

if ~exist('gui_dictprob')
   h = msgbox('Did you forget to export the dictionary? Run the section again, and remember to press [e] before closing the GUI.') 
   waitfor(h)
else
    dictionary = update_dictionary(dictionary,gui_dictprob); %update dictionary to include the learnt probalilities
    %USER INPUT: Save dictionary to process the complete scan
    uisave('dictionary',path_saveDict)
end