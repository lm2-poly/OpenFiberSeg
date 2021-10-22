function [V,xResolution,yResolution] = loadVolumeRoI_modFacu(path,FOV_handle,rectFoV,depth)
%LOADVOLUMEROI Loads a selected region of interest from a volume.
%   Details...
%by Monica Jane Emerson, March 2019, MIT License

contents_datafolder = dir(path);
path_firstSlice = [path,contents_datafolder(3).name];

%select FoV in the cross-sectional direction
if FOV_handle
    rect = rectFoV;
else
    rect = [1  1  size(imread(path_firstSlice),2) size(imread(path_firstSlice),1)];
           %y1 x1           yEnd                          xEnd 
end

%select slices

num_slices = length(contents_datafolder)-2; %depth of the volume
zmin = ['Lower limit in the depth direction [1, ',num2str(num_slices),']:'];
zmax = ['Upper limit in the depth direction [1, ',num2str(num_slices),']:'];
zstep = ['Step (1 to take every slice):'];
prompt = {zmin,zmax,zstep};

slices = depth(1):depth(3):depth(2);

V = zeros([rect(4)-rect(2)+1,rect(3)-rect(1)+1,numel(slices)],'uint16'); %volume preallocation (for speed)

tstart = tic;
h = waitbar(0,'Loading volume...', 'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
 for k= 1:numel(slices)
     
     if k==1
        %get scaling of original image, so processed data is also scaled
         tiffImInfo=imfinfo([path,contents_datafolder(slices(k)+2).name]);
        
         xResolution=tiffImInfo.XResolution;
         yResolution=tiffImInfo.YResolution;
    end
    im = imread([path,contents_datafolder(slices(k)+2).name]);
    V(:,:,k) = im2uint16(im(rect(2):rect(4),rect(1):rect(3)));
    
    %Processing data wait bar
    waitbar(k/numel(slices),h)
    if getappdata(h,'canceling')
        break
    end
    if k == numel(slices) 
        delete(h)
        t_load = toc(tstart);
        %f = msgbox(['time to load volume: ' , num2str(t_load), 's']);
    end
 end
% waitfor(f)
end

