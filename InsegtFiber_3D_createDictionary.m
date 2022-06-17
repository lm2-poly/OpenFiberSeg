% InSegtFibre_3D
% Written by Monica Jane Emerson, May 2018.
% MIT license
% Appended by Facundo Sosa-Rey, June 2019

%% Step 0: Add paths of Insegt GUI, other functions and scripts, and prepare workspace

close all,
clear
addpath(genpath('./InsegtLibraries/texture_gui'))
addpath('./InsegtLibraries/scripts')
addpath('./InsegtLibraries/functions')

fprintf('Step 0 completed\n')

% Step 1: Locate folder with the CT data
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% scanName='PEEK15/';
% scanName='PEEK20/';
% scanName='PEEK25/';
% scanName='PEEK30/';
% scanName='PEEK40/';

scanName=char(scanName);

switch scanName
        
    case 'PEEK15/'
        commonPath=  ['./TomographicData/'];
                
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        
    case 'PEEK20/'
        commonPath=  ['./TomographicData/'];
        
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
                
    case 'PEEK25/'
        commonPath=  ['./TomographicData/'];
                
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=976;
        
    case 'PEEK30/'
        commonPath=  ['./TomographicData/'];
        
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        
    case 'PEEK40/'
        commonPath=  ['./TomographicData/'];
                       
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=10; 
 
end

preprocessedFiles='preProcessed/';

path_volumeFolderHist= [commonPath scanName preprocessedFiles 'V_hist.tiff'];

fprintf('Loading data...\n')

% Step 2: Load and visualise the data
close all

rectFoV=[yMin xMin yMax xMax];

%hard-coded slice selection
depth=zeros(1,3);
depth(1)=zMin; %start slice
depth(2)=zMax; %end slice
depth(3)=1;%step size

totalFoV=[rectFoV([2,1]) depth(1); rectFoV([4,3]) depth(2)];

[V_hist,~,~,~] = loadVolumeRoI_modFacu_Volumetric(path_volumeFolderHist,rectFoV,depth);

% V_hist=permute(V_hist,[1 3 2]);

h0=figure; imshow3D(V_hist) %visualise the loaded RoI

uiwait(h0)

% Step 3: Compute a dictionary model from the CT scan

path_saveDict=[commonPath scanName preprocessedFiles 'dictionary.mat'];

setup_dict %set the parameters of the dictionary model
create_dict %learn the dictionary model from training data
%
fprintf('Dictionary created\n')

% Step 4: Obtain the centre-class probabilistic segmentation for a slice
close all

% MANUAL USER INPUT: Choose slice
prompt = ['Choose a slice to segment: [1,', num2str(size(V_hist,3)),']:'];
x = inputdlg(prompt,'User input',[1,30],{num2str(round(size(V_hist,3)/4))});
sliceAnalys = str2double(x{1});

fprintf('Segmenting sample image...\n');
[~,allP] = process_image(V_hist(:,:,sliceAnalys),dictionary);

Pcentre = allP(:,:,2);

%Visualise for each pixel its probabily of belonging to a fibre centre region
h = figure(1); h.Units = 'normalized'; h.Position = [0.1 0.1 .9 .9];
subplot(1,2,1), imagesc(Pcentre), axis image, colormap gray,  colorbar,
figure(1), subplot(1,3,3), histogram(Pcentre(:)),
ylabel('Count of probabilities'), xlabel('Range of probabilities')
suptitle('Probability of belonging to the central region of a fibre')

uiwait(h)

%Threshold probability map
% Step 5: Obtain the fibre centres by thresholding the probabilistic segmentation
% USER INPUT: threshold value for the probability map 
thresh = inputdlg('Choose threshold: ','User input',[1,20],{num2str(0.50)});

areas_th = Pcentre > str2double(thresh{1});

tempImFibers=ind2rgb(areas_th, [0 0 0; 255, 100, 255]/255);
alphaParam=0.7;
tempImHistRGB=double(cat(3, V_hist(:,:,sliceAnalys), V_hist(:,:,sliceAnalys), V_hist(:,:,sliceAnalys)))/255;
tempImOutput=tempImHistRGB*(1-alphaParam)+tempImFibers*alphaParam;

figure(1)
imshowpair(areas_th,V_hist(:,:,sliceAnalys),'montage')
title('segmentation (left) vs original data (right)','fontsize',16)

figure(2)
imshow(tempImOutput)
title('segmentation overlaid on original data','fontsize',16)

fprintf('Done\n')
