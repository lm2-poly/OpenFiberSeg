function [V,depth,rect] = loadVolumeRoI(path,FOV_handle,slices_handle,depth)
%LOADVOLUMEROI Loads a selected region of interest from a volume.
%   Details...
%by Monica Jane Emerson, March 2019, MIT License

contents_datafolder = dir(path);
path_firstSlice = [path,contents_datafolder(3).name];

%select FoV in the cross-sectional direction
if FOV_handle
    [rect,~] = selectFoV(path_firstSlice);
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

if slices_handle
    depth = inputdlg(prompt,'RoI in the depth direction', [1,50], {'1',num2str(num_slices),'10'});
end

slices = [str2double(depth{1}):str2double(depth{3}):str2double(depth{2})];

V = zeros([rect(4)-rect(2)+1,rect(3)-rect(1)+1,numel(slices)]); %volume preallocation (for speed)

tstart = tic;
h = waitbar(0,'Loading volume...', 'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
 for k= 1:numel(slices)
    im = imread([path,contents_datafolder(slices(k)+2).name]);
    V(:,:,k) = double(im(rect(2):rect(4),rect(1):rect(3)))/2^16;
    
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

