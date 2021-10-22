function [V,xResolution,yResolution,unitTiff] = loadVolumeRoI_modFacu_Volumetric(path,rect,depth)
%LOADVOLUMEROI Loads a selected region of interest from a volume.
%   Details...
%by Monica Jane Emerson, March 2019, MIT License

tiffImInfo=imfinfo(path);
nSlices=numel(tiffImInfo);


%selected slices
slices = depth(1):depth(3):depth(2);

if numel(slices)>nSlices || depth(2)>nSlices
   error('slice values are outside the dimension of Tiff file, nSlices= %5.d ',nSlices);
end

% initialization
V = zeros([rect(4)-rect(2)+1,rect(3)-rect(1)+1,numel(slices)],'uint8'); %volume preallocation (for speed)

tstart = tic;
h = waitbar(0,'Loading volume...', 'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
for k= 1:numel(slices)
     
    if k==1
        %get scaling of original image, so processed data is also scaled
        
         xResolution=tiffImInfo(1).XResolution;
         yResolution=tiffImInfo(1).YResolution;
         unitTiff=tiffImInfo(1).ResolutionUnit;
    end
    
    im = imread(path,slices(k));
        
    V(:,:,k) = im2uint8(im(rect(2):rect(4),rect(1):rect(3)));
    
    %Processing data wait bar
    waitbar(k/numel(slices),h)
    if getappdata(h,'canceling')
        break
    end
    if k == numel(slices) 
        delete(h)
        t_load = toc(tstart);
        fprintf(['time to load volume: ' , num2str(t_load), 's\n']);
    end
 end

end

