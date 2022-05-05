% InSegtFibre_3D
% Written by Monica Jane Emerson, May 2018.
% MIT license
% Modifed by Facundo Sosa-Rey, June 2021

%% Step 0: Add paths of Insegt GUI, other functions and scripts, and prepare workspace

close all,
clear
addpath(genpath('./InsegtLibraries/texture_gui'))
addpath('./InsegtLibraries/scripts')
addpath('./InsegtLibraries/functions')
% parpool

myCluster = parcluster('local'); %get max number of workers on local machine
nPool=myCluster.NumWorkers-1;

%parfeval need a parallel pool
if isempty(gcp('nocreate'))
    parpool(nPool);
end

fprintf('Step 0 completed\n')


% Step 1: Locate folder with the CT data
close all

%indicate_dataFolder
dataPath='./TomographicData/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scanName='PEEK05/';
% scanName='PEEK10/';
% scanName='PEEK15/';
% scanName='PEEK20/';
% scanName='PEEK25/';
% scanName='PEEK30/';
% scanName='PEEK35/';
% scanName='PEEK40/';

scanName=char(scanName);

if scanName(end)~='/'
    scanName=[scanName '/'];
end

switch scanName
    case 'PEEK05/'
              
        preprocessedFiles='preProcessed/';
        
        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK05.mat';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=980;
        thresh{1}='0.508';%probability threshold value
                
    case 'PEEK10/'
        
        preprocessedFiles='preProcessed/';
        
        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK05.mat';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=980;
        thresh{1}='0.4995';%probability threshold value
                
    case 'PEEK15/'

        preprocessedFiles='preProcessed/';
  
        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK15.mat';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.500'; %probability threshold value
                
    case 'PEEK20/'

        preprocessedFiles='preProcessed/';

        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK20.mat';

        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.497'; %probability threshold value
                
    case 'PEEK25/'

        preprocessedFiles='preProcessed/';
     
        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK25.mat';
        
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=976;
        thresh{1}='0.499'; %probability threshold value
                
    case 'PEEK30/'

        preprocessedFiles='preProcessed/';

        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK30.mat';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.499'; %probability threshold value
                
    case 'PEEK35/'

        preprocessedFiles='preProcessed/';

        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK40.mat';%thresh{1}='0.502';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;

        thresh{1}='0.502'; %probability threshold value
                
    case 'PEEK40/'

        preprocessedFiles='preProcessed/';
        
        dictionaryFolder='InsegtDictionaryFiles/';
        dictionaryFile='dictionary_PEEK40.mat';%thresh{1}='0.500';
        path_to_dict=          [dataPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.498'; %probability threshold value
                
    
end

path_volumeFolderHist= [dataPath scanName preprocessedFiles 'V_hist.tiff'];

fprintf('Loading data...\n')

% Step 2: Load and visualise the data
close all
FOV_handle=true;%if FOV is passed as input argument, false if entire image is the FoV
% hard coded FoV

writeToDisk=true;
compressWithPython=true; % lossless compression with python module tifffile
% yields files twice as small. this step is optional, the process is a bit 
% long but halves the required disk space

timeStamp=datestr(now,'yyyy-mm-dd_HH:MM:SS');

timeStamp(14)='h';
timeStamp(17)='m';

permutationVecAll={[1 2 3], [1 3 2], [3 2 1]};

rectFoV=[yMin xMin yMax xMax];

%hard-coded slice selection
depth=zeros(1,3);
depth(1)=zMin; %start slice
depth(2)=zMax; %end slice
depth(3)=1;%step size

totalFoV=[rectFoV([2,1]) depth(1); rectFoV([4,3]) depth(2)];

[V_hist,xResolution,yResolution,unitTiff] = loadVolumeRoI_modFacu_Volumetric(path_volumeFolderHist,rectFoV,depth);

V_fibers = zeros(size(V_hist),'uint8');

V_prob=zeros(size(V_hist));

%load the porosity mask obtained in pre-processsing

%path to porosity mask
path_toPorosityMask=[dataPath scanName preprocessedFiles 'V_pores.tiff'];

[V_pores,~,~,~] = loadVolumeRoI_modFacu_Volumetric(path_toPorosityMask,rectFoV,depth);

%path to perimeter mask
path_toPerim=[dataPath scanName preprocessedFiles 'V_perim.tiff'];


if exist(path_toPerim,'file')==2
    perim_present=true;
    [V_perim,~,~,~] = loadVolumeRoI_modFacu_Volumetric(path_toPerim,rectFoV,depth);
else
    perim_present=false;    
    % needs to be initialized even if not used or parallel execution will crash
    V_perim=false(size(V_hist));
end



for iPermutation=1:3
    permutationVec=permutationVecAll{iPermutation};
    
    permutationStr=sprintf('Permutation%1d%1d%1d',permutationVec);
    fprintf(['Processing: ' permutationStr '... \n'])
    
    
    
    if all(permutationVec==[1 2 3])
        permuteBool=false;
    else
        permuteBool=true;
    end
    
    if permuteBool
        
        if isequal(permutationVec,[3,2,1])
           permutationVec=[2,3,1]; %since it was permuted from 123 to 132 beforehand
        end
        
        totalFoVnew=totalFoV(:,permutationVec);
        rectFoV=[totalFoV(1,2),totalFoV(1,1),totalFoV(2,2),totalFoV(2,1)];
                
        V_hist=permute(V_hist,permutationVec);
        V_pores=permute(V_pores,permutationVec);
        V_fibers=permute(V_fibers,permutationVec);
        V_prob=permute(V_prob,permutationVec);
        V_perim=permute(V_perim,permutationVec);
        
    end
    
    figure, imshow3D(V_hist) %visualise the loaded RoI
    figure, imshow3D(V_pores) %visualise the loaded mask RoI
    
    fprintf('Permutation completed\n')
    
    % Step 3: Compute a dictionary model from the CT scan
    close all
    path_volumeFolder=path_volumeFolderHist;
    
%     setup_dict %set the parameters of the dictionary model
%     create_dict %learn the dictionary model from training data
    %
%     fprintf('Step 3 completed')
    
    % Step 4: Obtain the centre-class probabilistic segmentation for a slice
    close all
    
    % hard-coded dictionary location
    load(path_to_dict);% redo from right location
    
    %hard-coded slice to segment: midpoint
    nextSlice=round(size(V_hist,3)/2);
    
    fprintf('Segmenting sample image...\n');
    [~,allP] = process_image(V_hist(:,:,nextSlice),dictionary);
    
    Pcentre = allP(:,:,2);
    %Store Probability map
    V_prob(:,:,nextSlice)=Pcentre;
    
    %Visualise for each pixel its probabily of belonging to a fibre centre region
    h = figure(1); h.Units = 'normalized'; h.Position = [0.1 0.1 .9 .9];
    subplot(1,2,1), imagesc(Pcentre), axis image, colormap gray,  colorbar,
    figure(1), subplot(1,2,2), histogram(Pcentre(:)),
    ylabel('Count of probabilities'), xlabel('Range of probabilities')
    suptitle('Probability of belonging to the central region of a fibre')
    
    %Threshold probability map
    areas_th = Pcentre > str2double(thresh{1});
    
    tempImPores=ind2rgb(areas_th, [0 0 0; 255, 100, 255]/255);
    alphaParam=0.7;
    tempImHistRGB=double(cat(3, V_hist(:,:,nextSlice), V_hist(:,:,nextSlice), V_hist(:,:,nextSlice)))/255;
    tempImOutput=tempImHistRGB*(1-alphaParam)+tempImPores*alphaParam;
    
    figure(2);
    imshowpair(areas_th,V_hist(:,:,nextSlice),'montage')
    title('segmentation (left) vs original data (right)','fontsize',16)
    figure(3)
    imshow(tempImOutput)
    title('segmentation overlaid on original data','fontsize',16)
    
    uiwait(h,60)
    
    % Step 5: Segmentation of entire volume
    fprintf('Segmentation of entire volume...\n')
    
    close all
    clear allP
    
    %Initialise variables
    step = 1;% allows the skipping of steps
    slices = 1:step:size(V_hist,3);
    centrePoints = [];
    
    %Start timer
    tstart = tic;
    % h = waitbar(0,'Processing volume...');
    
    %Process the volume
    
    parfor nextSlice = 1:numel(slices)
        %for iSlice = 1:numel(slices_frag)
        %Compute the probabilistic segmentation
        [~,allP] = process_image(V_hist(:,:,nextSlice),dictionary);
        
        %store probability map
        V_prob(:,:,nextSlice)=allP(:,:,2);
        
        %store segmentation data
        logicalSegt=allP(:,:,2) > str2num(thresh{1});
        
        %regions labeled as pores (V_pores==true) cant be labelled as fibres
        logicalSegt(logical(V_pores(:,:,nextSlice)))=false;
        if perim_present
            %regions labeled as perimeter (V_perim==true) cant be labelled as fibres
            logicalSegt(logical(V_perim(:,:,nextSlice)))=false;
        end
        V_fibers(:,:,nextSlice)=im2uint8(logicalSegt);
        
    end
    
    
    if writeToDisk
        
        
        strFoV=sprintf('_x%.d-%.d_y%.d-%.d_z%.d-%.d',xMin,xMax,yMin,yMax,zMin,zMax);
        
        [xDim,yDim,zDim]=size(V_hist);
        
        descriptionStr=sprintf('{"shape ([x,y,z])":[%4.d, %4.d, %4.d]}',xDim,yDim,zDim);
        
        
        savePath=[dataPath scanName FolderTag strFoV '/' timeStamp '/' permutationStr ];
        
        fprintf(['Writing to: \n' savePath '\n'])
        
        mkdir(savePath)
        
        V_export_nameHist=[savePath '/V_hist.tiff'];
        V_export_namePores=[savePath '/V_pores.tiff'];
        V_export_nameSegt=[savePath '/V_fibers.tiff'];
        V_export_nameProb=[savePath '/V_prob.tiff'];
        if perim_present
            V_export_namePerim=[savePath '/V_perim.tiff'];
        end
        
        for iSlice=1:length(V_hist(1,1,:))
            if iSlice==1
                if strcmp(unitTiff,'Centimeter')
                    scaleFactor=2.54;
                elseif strcmp(unitTiff,'Inch')
                    scaleFactor=1;
                end
                imwrite(im2uint8(V_hist(:,:,iSlice)),V_export_nameHist,...
                    'Resolution',[xResolution,yResolution]*scaleFactor,...
                    'Description',descriptionStr); %only 16 discrete bins do to hist eq, no need for uint16. this data is only used for visual validation, not processing
                imwrite(im2uint8(V_pores(:,:,iSlice)),   V_export_namePores,...
                    'Resolution',[xResolution,yResolution]*scaleFactor,...
                    'Description',descriptionStr);
                imwrite(V_fibers(:,:,iSlice),     V_export_nameSegt,...
                    'Resolution',[xResolution,yResolution]*scaleFactor,...
                    'Description',descriptionStr);
                imwrite(im2uint8(V_prob(:,:,iSlice)),     V_export_nameProb,...
                    'Resolution',[xResolution,yResolution]*scaleFactor,...
                    'Description',descriptionStr);
                if perim_present
                    imwrite(V_perim(:,:,iSlice),     V_export_namePerim,...
                        'Resolution',[xResolution,yResolution]*scaleFactor,...
                    'Description',descriptionStr);              
                end
            else
                imwrite(im2uint8(V_hist (:,:,iSlice)),V_export_nameHist, 'WriteMode', 'append');
                imwrite(im2uint8(V_pores(:,:,iSlice)),V_export_namePores,'WriteMode', 'append');
                imwrite(V_fibers(:,:,iSlice),         V_export_nameSegt, 'WriteMode', 'append');
                imwrite(im2uint8(V_prob(:,:,iSlice)),V_export_nameProb, 'WriteMode', 'append');
                if perim_present
                    imwrite(V_perim(:,:,iSlice),     V_export_namePerim,'WriteMode', 'append');              
                end
            end
        end
        
        if iPermutation==1
            exportData.V_export_name=V_export_nameHist;
            exportData.scanName     =scanName;
            exportData.rectFoV          =rectFoV;
            exportData.xMin             =xMin;
            exportData.xMax            =xMax;
            exportData.yMin             =yMin;
            exportData.yMax             =yMax;
            exportData.zMin             =zMin;
            exportData.zMax             =zMax;
            
            exportData.dictionaryFolder=path_to_dict;
            exportData.dictionaryFile=dictionaryFile;
            exportData.probabilityThresh=thresh{1};
            exportData.preProcessedFiles=preprocessedFiles;
            exportData.dataPath=dataPath;
            exportData.permutationVec=permutationVec;
            
            jsonContent=jsonencode(exportData);
            
            savePathParam=[dataPath scanName FolderTag strFoV '/' timeStamp '/'];
            
            filename = [savePathParam '/SegtParams' ];
            
            fid = fopen([filename '.json'],'w');    % open file for writing (overwrite if necessary)
            fprintf(fid,jsonContent);          % Write the char array, interpret newline as new line
            fclose(fid);
        end        
        
        if compressWithPython
            % higher (lossless) compression rate is achieved with python module tifffile 
            system(['python3 compressTiff.py "' savePath '/"'])
        end
        
    end
    
end


fprintf('Done\n')


