function [V_mask] = loadVolumeMask(path,rect,depth)
%LOADVOLUMEMASK Loads the porosity mask for the selected region of interest from a volume.
%by Monica Jane Emerson, March 2019, MIT License
% Modified by Facundo Sosa-Rey, June 2019

contents_datafolder = dir(path);
path_firstSlice = [path,contents_datafolder(3).name];

%FoV in the cross-sectional direction
%rect variable is obtained in loadVolumeROI()

%select slices
num_slices = length(contents_datafolder)-2; %depth of the volume
zmin = ['Lower limit in the depth direction [1, ',num2str(num_slices),']:'];
zmax = ['Upper limit in the depth direction [1, ',num2str(num_slices),']:'];
zstep = ['Step (1 to take every slice):'];
prompt = {zmin,zmax,zstep};
%depth passed as argument, obtained previously
%depth = inputdlg(prompt,'RoI in the depth direction', [1,50], {'1',num2str(num_slices),'10'});
slices = [str2double(depth{1}):str2double(depth{3}):str2double(depth{2})];

V_mask = false([rect(4)-rect(2)+1,rect(3)-rect(1)+1,numel(slices)]); %volume preallocation (for speed)

tstart = tic;
h = waitbar(0,'Loading porosity mask...', 'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
 for k= 1:numel(slices)
    im = imread([path,contents_datafolder(slices(k)+2).name]);
    V_mask(:,:,k) = im(rect(2):rect(4),rect(1):rect(3));
    
    %Processing data wait bar
    waitbar(k/numel(slices),h)
    if getappdata(h,'canceling')
        break
    end
    if k == numel(slices) 
        delete(h)
        t_load = toc(tstart);
        %f = msgbox(['time to load porosity mask: ' , num2str(t_load), 's']);
    end
 end
% waitfor(f)
end

