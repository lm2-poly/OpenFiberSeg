% InSegtFibre_3D
% Written by Monica Jane Emerson, May 2018.
% MIT license
% Modified by Facundo Sosa-Rey, June 2019

%% Step 0: Add paths of Insegt GUI, other functions and scripts, and prepare workspace

close all,
clear
addpath(genpath('./texture_gui'))
addpath('./scripts')
addpath('./functions')
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

%manual input
%indicate_dataFolder
name=convertCharsToStrings(getComputerName());
nameArray=["facundo-thinkpad-kubuntu19","facu-x299-kubuntu20.04"];

if strcmp(name,nameArray(1))
    dataPath='/media/facu/DataNTFS_Crucial/TomographicData/';
elseif strcmp(name,nameArray(2))
%     commonPath=  '/media/facu/DataX299_nvme/TomographicData/';
    dataPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% scanName='tiff files-peek40-3Dprint_2019-02-25/';
% scanName='markforgedFilament20x_2020-08-22/';
% scanName='8.8_pixelsDavg_0/';
% scanName='8.8_pixelsDavg_+-45/';
% scanName='8.8_pixelsDavg_+45-60/';

% scanName='TIFF LM PEEK 05/';
% scanName='TIFF LM PEEK 10/';
% scanName='TIFF LM PEEK 15/';
% scanName='TIFF LM PEEK 20/';
% scanName='TIFF LM PEEK 25/';
% scanName='TIFF LM PEEK 30/';
% scanName='TIFF LM PEEK 35/';
% scanName='TIFF LM PEEK 40/';

% scanName='TIFF LM PEEKmilled20/';
% scanName='TIFF LM PEEKmilled40/';

% scanName='TIFF LM PEI 05/';
% scanName='TIFF LM PEI 10/';
% scanName='TIFF LM PEI 15/';
% scanName='TIFF LM PEI 20/';
% scanName='TIFF LM PEI 25/';
% scanName='TIFF LM PEI 30/';

% scanName='Czabaj_Unidirectional/';
scanName='Czabaj_Unidirectional_for article/';

% scanName='Czabaj_Multi-directional/';

% scanName='Badran_Processed_RECON_20161026_212323_HNS-MUC-03_mono_10X_baseline_tile_x00y05/';

% scanName='DogboneInjectedPEEK_CF/';

% scanName="QIM_structureTensorData/";

% scanName='DogboneInjected_8230_segment/';

% scanName="Blend80-20_3Dprint50kv4W_10x/";

% scanName='TiffFilesYahya11oct 8230Z-B2-extrusion_2019-10-11_120052/';

% scanName='InjectedDogbone_type1_8230_sample_A_2021-06-29_175728';
% scanName='InjectedDogbone_type1_8230_sample_B_2021-06-29_111850/';
% scanName='Injected8230Z_Dogbone_SampleC/';

% scanName='Paul_8230Z_Filament175_printedStrand_2021-06-28_111912/';
% scanName='Yahya_8230Z_printedStrand_2021-06-16_152654/';

scanName=char(scanName);

if scanName(end)~='/'
    scanName=[scanName '/'];
end

switch scanName
    case 'tiff files-peek40-3Dprint_2019-02-25/'
        
        preprocessedFiles='preProcessed_Oct-21-2020/'; 
        dictionaryFolder='dictionairy 13-Jun-2019/';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        
%         % inclined large fiber
%         FolderTag='InclinedLargeFiber';
%         xMin=210+160;
%         xMax=440+200;
%         yMin=270+160;
%         yMax=520+200;
%         zMin=550; % 550-50;
%         zMax=630; %630+170;
%         thresh{1}='0.498'; %probability threshold value
%           
%         FolderTag='CombinePermutationTest';
%         xMin=210+160;
%         xMax=440+200;
%         yMin=270+160;
%         yMax=520+200;
%         zMin=550-50;
%         zMax=630+170;
%         thresh{1}='0.498'; %probability threshold value
%         
%         FolderTag='mismatchZoom';
%         xMin=280;
%         xMax=410;
%         yMin=230;
%         yMax=380;
%         zMin=550;
%         zMax=610;
%         thresh{1}='0.498'; %probability threshold value
        
%         FolderTag='LargeRegion';
%         xMin=1;
%         xMax=652;
%         yMin=1;
%         yMax=652;
%         zMin=1;
%         zMax=977;
%         thresh{1}='0.498'; %probability threshold value   
        
%         FolderTag='LargeRegion8Bit';
%         xMin=1;
%         xMax=652;
%         yMin=1;
%         yMax=652;
%         zMin=1;
%         zMax=10;
%         thresh{1}='0.498'; %probability threshold value  
        
        FolderTag='FullVolume';
        xMin=1;
        xMax=1001;    
        yMin=1;
        yMax=956;
        zMin=1;
        zMax=977;
        thresh{1}='0.498'; %probability threshold value  
                
    case 'markforgedFilament20x_2020-08-22/'
        
        % preprocessedFiles='Processed_22-Aug-2020/';
%         preprocessedFiles='Processed_23-Aug-2020/';
%         preprocessedFiles='preProcessed_17-Sep-2020/';
        preprocessedFiles='preProcessed_Oct-16-2020/'; % Volumetric tiff
        dictionaryFolder='dictionairy 13-Jun-2019/';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        
        % markedForge
        FolderTag='markForged_EntireVolume';
%         xMin=1;
%         xMax=752;
%         yMin=1;
%         yMax=782;
%         zMin=1;
%         zMax=300;
%         thresh{1}='0.498'; %probability threshold value
        
        xMin=1;
        xMax=831;
        yMin=1;
        yMax=861;
        zMin=1;
        zMax=40;%965;
        thresh{1}='0.498'; %probability threshold value
                
    case '8.8_pixelsDavg_0/'
        commonPath=  [dataPath 'Czabaj/microCT_Czabaj/SyntheticData/Images/']
        
%         preprocessedFiles='Processed_24-Aug-2020/';
        preprocessedFiles='preProcessed_Jun-14-2021/';
        dictionaryFolder='dictionary 24-Aug-2020/';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath scanName dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=146;
        yMin=1;
        yMax=301;
        zMin=1;
        zMax=277;
        thresh{1}='0.498'; %probability threshold value
                
    case '8.8_pixelsDavg_+-45/'
        commonPath=  [dataPath 'Czabaj/microCT_Czabaj/SyntheticData/Images/']
        
%         preprocessedFiles='Processed_24-Aug-2020/';
        preprocessedFiles='preProcessed_Jun-14-2021/';
        dictionaryFolder='dictionary 24-Aug-2020/';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath scanName dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=120;
        yMin=1;
        yMax=379;
        zMin=1;
        zMax=276;
        thresh{1}='0.498'; %probability threshold value
                
    case '8.8_pixelsDavg_+45-60/'
        commonPath=  [dataPath 'Czabaj/microCT_Czabaj/SyntheticData/Images/']
        
%         preprocessedFiles='Processed_24-Aug-2020/';
        preprocessedFiles='preProcessed_Jun-14-2021/';
        dictionaryFolder='dictionary 24-Aug-2020/';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath scanName dictionaryFolder dictionaryFile];
        
        %SyntheticData
        FolderTag='processed';
        xMin=1;
        xMax=120;
        yMin=1;
        yMax=379;
        zMin=1;
        zMax=257;
        thresh{1}='0.498'; %probability threshold value
             
    case 'TIFF LM PEEK 05/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
        dictionaryFolder='dictionairy 13-Jun-2019/';
        dictionaryFile='dictionary_v_PEEK_05.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=980;
        thresh{1}='0.508';%'0.500'; %probability threshold value
                
    case 'TIFF LM PEEK 10/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
        dictionaryFolder='dictionairy 13-Jun-2019/';
        dictionaryFile='dictionary_v_PEEK_05.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=980;
        thresh{1}='0.4995';%'0.500'; %probability threshold value
                
    case 'TIFF LM PEEK 15/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500';
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEK15.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.500'; %probability threshold value
                
    case 'TIFF LM PEEK 20/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v_PEEK_05.mat';thresh{1}='0.500';

        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEK20.mat';

        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.497'; %probability threshold value
                
    case 'TIFF LM PEEK 25/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500';
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEK25.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=976;
        thresh{1}='0.499'; %probability threshold value
                
    case 'TIFF LM PEEK 30/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500';
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEK30.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.499'; %probability threshold value
                
    case 'TIFF LM PEEK 35/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
        dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500';
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEK40_v2.mat';%thresh{1}='0.502';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;

        thresh{1}='0.502'; %probability threshold value
                
    case 'TIFF LM PEEK 40/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';thresh{1}='0.4985'
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
%         dictionaryFile='dictionary_PEEK40.mat';%thresh{1}='0.500';
        dictionaryFile='dictionary_PEEK40_v2.mat';%thresh{1}='0.500';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.498'; %probability threshold value
                
    case 'TIFF LM PEEKmilled20/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500';
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEKmilled20.mat';%thresh{1}='0.500';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.500'; %probability threshold 
                
    case 'TIFF LM PEEKmilled40/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEEK/'];
        
        preprocessedFiles='preProcessed_Oct-16-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';
%         dictionaryFile='dictionary_v_PEEK_05.mat';%thresh{1}='0.500'; 
        
        dictionaryFolder='dictionairy 07-Mai-2021/';
        dictionaryFile='dictionary_PEEKmilled40.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.499'; %probability threshold value
                
    case 'TIFF LM PEI 05/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        preprocessedFiles='preProcessed_Oct-26-2020/';
        dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v_PEEK_05.mat';
        dictionaryFile='dictionary_v.mat';
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMCzabaj_Unidirectional_for article/in=1;
        xMax=901;
        yMin=1;
        yMax=911;
        zMin=1;
        zMax=978;
        thresh{1}='0.4995'; %probability threshold value
        
    case 'TIFF LM PEI 10/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        preprocessedFiles='preProcessed_Oct-26-2020/';
        dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v_PEEK_05.mat';
        dictionaryFile='dictionary_v.mat';

        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.499'; %probability threshold value
                
    case 'TIFF LM PEI 15/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        prCzabaj_Unidirectional_for article/eprocessedFiles='preProcessed_Oct-26-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v_PEEK_05.mat';
%         dictionaryFile='dictionary_v.mat';%thresh{1}='0.499';
        
        dictionaryFolder='dictionairy 11-mai-2021/';
        dictionaryFile='dictionary_PEI15.mat';
        
        

        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=978;
        thresh{1}='0.499'; %probability threshold value
               
    case 'TIFF LM PEI 20/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        preprocessedFiles='preProcessed_Oct-26-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';thresh{1}='0.498'
        
        dictionaryFolder='dictionairy 11-mai-2021/';
        dictionaryFile='dictionary_PEI20.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=979;
        thresh{1}='0.498'; %probability threshold value
                
    case 'TIFF LM PEI 25/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        preprocessedFiles='preProcessed_Oct-26-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';thresh{1}='0.498';
        
        dictionaryFolder='dictionairy 11-mai-2021/';
        dictionaryFile='dictionary_PEI25.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=977;
        thresh{1}='0.496'; %probability threshold value
                
    case 'TIFF LM PEI 30/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Loic/ToTransferComputeCanada/PEI/'];
        
        preprocessedFiles='preProcessed_Oct-26-2020/';
%         dictionaryFolder='dictionairy 13-Jun-2019/';
%         dictionaryFile='dictionary_v.mat';thresh{1}='0.497'
        
        dictionaryFolder='dictionairy 11-mai-2021/';
        dictionaryFile='dictionary_PEI30.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=901;
        yMin=1;
        yMax=871;
        zMin=1;
        zMax=977;
        thresh{1}='0.4985'; %probability threshold value
        
    case 'Czabaj_Unidirectional/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Czabaj/microCT_Czabaj/'];
        
        preprocessedFiles='preProcessed_Dec-07-2020/';
        dictionaryFile='dictionaryPatch17_v2.mat';
        
        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=426;
        yMin=1;
        yMax=426;
        zMin=1;
        zMax=426;
        thresh{1}='0.51'; %probability threshold value
    case 'Czabaj_Unidirectional_for article/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Czabaj/microCT_Czabaj/'];
        
        preprocessedFiles='preProcessed_Oct-16-2021/';
        dictionaryFile='dictionaryPatch17_v2.mat';
        
        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=426;
        yMin=1;
        yMax=426;
        zMin=1;
        zMax=426;
        thresh{1}='0.51'; %probability threshold value
        
        
    case 'Czabaj_Multi-directional/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Czabaj/microCT_Czabaj/'];
        
        preprocessedFiles='preProcessed_Dec-14-2020/';
        dictionaryFile='dictionaryPatch17_v2.mat';

        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=501;
        yMin=1;
        yMax=501;
        zMin=1;
        zMax=601;
        thresh{1}='0.51'; %probability threshold value
        
    case 'Badran_Processed_RECON_20161026_212323_HNS-MUC-03_mono_10X_baseline_tile_x00y05/'
        commonPath=  ['/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Badran/TestTransferGlobus/Recon/'];
        
        preprocessedFiles='preProcessed_Jan-07-2021/';
%         dictionaryFile='dictionaryPatch21.mat';
%         thresh{1}='0.498'; %probability threshold value

        dictionaryFile='dictionaryPatch27_centersOnly.mat';
        thresh{1}='0.5'; %probability threshold value

        
        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=1621;
        yMin=1;
        yMax=1601;
        zMin=1;
        zMax=2159;
        
        
    case 'DogboneInjectedPEEK_CF/'
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/';
        preprocessedFiles='preProcessed_Feb-05-2021/'; 

        dictionaryFile='dictionary.mat';
        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
%         FolderTag='processed';
%         xMin=1;
%         xMax=1009;    
%         yMin=1;
%         yMax=987;
%         zMin=1;
%         zMax=991;
%         thresh{1}='0.495'; %probability threshold value  

        FolderTag='partialVolume';
        xMin=300;
        xMax=600;    
        yMin=250;
        yMax=700;
        zMin=1;
        zMax=600;
        thresh{1}='0.495'; %probability threshold value  
        
    case 'QIM_structureTensorData/'

        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/';
        preprocessedFiles='preProcessed_Feb-24-2021/'; 

        dictionaryFile='dictionary_132.mat';
        path_to_dict=          [commonPath scanName preprocessedFiles dictionaryFile];
        
        FolderTag='processed';
        xMin=1;
        xMax=320;    
        yMin=1;
        yMax=320;
        zMin=1;
        zMax=320;
        thresh{1}='0.495'; %probability threshold value  
        
    case "DogboneInjected_8230_segment/"
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/'
        preprocessedFiles='preProcessed_May-11-2021/';
        
        dictionaryFolder=scanName;
        dictionaryFile='dictionary_dogbone8230injected.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];
        
        FolderTag='processed';

        xMin=1;
        xMax=967;    
        yMin=1;
        yMax=944;
        zMin=1;
        zMax=975;  
        
        thresh{1}='0.499'; %probability threshold value  
        
    case "Blend80-20_3Dprint50kv4W_10x/"
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Audrey/Superstar8020/'
        preprocessedFiles='preProcessed_Jun-07-2021/';
        
        dictionaryFolder=scanName;
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';

        xMin=1;
        xMax=1001;    
        yMin=1;
        yMax=960;
        zMin=1;
        zMax=977;
        
        thresh{1}='0.500'; %probability threshold value  
        
    case 'TiffFilesYahya11oct 8230Z-B2-extrusion_2019-10-11_120052/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/'
        preprocessedFiles='preProcessed_Jun-10-2021/';
        
        dictionaryFolder=scanName;
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1001;    
        yMin=1;
        yMax=972;
        zMin=1;
        zMax=976;
        
        thresh{1}='0.499'; %probability threshold value  
        
    case 'InjectedDogbone_type1_8230_sample_A_2021-06-29_175728/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/InjectedDogbone8230Z/'
        preprocessedFiles='preProcessed_Jul-06-2021/';
        
        dictionaryFolder=[scanName preprocessedFiles];
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1003;
        yMin=1;
        yMax=974;
        zMin=1;
        zMax=978;
        
        thresh{1}='0.500'; %probability threshold value  

    case 'InjectedDogbone_type1_8230_sample_B_2021-06-29_111850/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/InjectedDogbone8230Z/'
        preprocessedFiles='preProcessed_Jul-07-2021/';

        dictionaryFolder=[scanName preprocessedFiles];
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1003;
        yMin=1;
        yMax=933;
        zMin=1;
        zMax=978;   
        
        thresh{1}='0.498'; %probability threshold value  

        
    case 'Injected8230Z_Dogbone_SampleC/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/InjectedDogbone8230Z/'
        preprocessedFiles='preProcessed_Jul-02-2021/';
        
        dictionaryFolder=[scanName preprocessedFiles];
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1003;
        yMin=1;
        yMax=974;
        zMin=1;
        zMax=975;
        
        thresh{1}='0.500'; %probability threshold value  

    case 'Paul_8230Z_Filament175_printedStrand_2021-06-28_111912/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/'
        preprocessedFiles='preProcessed_Jul-03-2021/';
        
        dictionaryFolder=[scanName preprocessedFiles];
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1003;
        yMin=1;
        yMax=957;
        zMin=1;
        zMax=976;

        thresh{1}='0.501'; %probability threshold value  
    case 'Yahya_8230Z_printedStrand_2021-06-16_152654/'
        
        commonPath='/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Yahya/'
        preprocessedFiles='preProcessed_Jul-03-2021/';
        
        dictionaryFolder=[scanName preprocessedFiles];
        dictionaryFile='dictionary.mat';
        
        path_to_dict=          [commonPath dictionaryFolder dictionaryFile];

        FolderTag='processed';
        xMin=1;
        xMax=1003;
        yMin=1;
        yMax=980;
        zMin=1;
        zMax=978;
        
        thresh{1}='0.503'; %probability threshold value  
end

path_volumeFolderHist= [commonPath scanName preprocessedFiles 'V_hist.tiff'];


% [contents_datafolder,path_firstSlice,im]=indicate_dataFolderFunction(path_volumeFolderHist);% change this!!!

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
path_toPorosityMask=[commonPath scanName preprocessedFiles 'V_pores.tiff'];

[V_pores,~,~,~] = loadVolumeRoI_modFacu_Volumetric(path_toPorosityMask,rectFoV,depth);

%path to perimeter mask
path_toPerim=[commonPath scanName preprocessedFiles 'V_perim.tiff'];


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
        
        
        savePath=[commonPath scanName FolderTag strFoV '/' timeStamp '/' permutationStr ];
        
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
                imwrite(im2uint16(V_prob(:,:,iSlice)),     V_export_nameProb,...
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
                imwrite(im2uint16(V_prob(:,:,iSlice)),V_export_nameProb, 'WriteMode', 'append');
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
            exportData.commonPath=commonPath;
            exportData.permutationVec=permutationVec;
            
            jsonContent=jsonencode(exportData);
            
            savePathParam=[commonPath scanName FolderTag strFoV '/' timeStamp '/'];
            
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


