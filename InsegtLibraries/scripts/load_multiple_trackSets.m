%LOAD_MULTIPLE_TRACKSETS is a user-friendly script to load tracks measured 
%from multiple data-sets, e.g. acquired at progressive load steps.
% by Monica Jane Emerson, March 2019, MIT License

%% USER INPUT: Load several sets of tracks corresponding to different steps of a time-lapse data-set
x = inputdlg('How many sets of tracks would you like to load [>=2]?','Nr. analysed matching volumes',[1,60],{num2str(2)});

for l = 1:str2double(x{1})
    [filename_tracks,pathname_tracks] = uigetfile('*.mat',['Choose tracks for data-set ',num2str(l)],'../data/saved_data/fibreTracks');
    aux = load([pathname_tracks,filename_tracks]);
    tracks{l} = aux.fibres;
end
num_datasets = length(tracks);
num_slices = size(tracks{1}.x,2);

