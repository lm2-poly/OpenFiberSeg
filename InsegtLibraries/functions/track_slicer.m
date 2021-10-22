function track_slicer(filename,centers,clim)
%TRACK_SLICER   Slice trough data, centers and tracks.
%   TRACK_SLICER(FILENAME,CENTERS,CLIM)
%       FILENAME, name of the tiff stack with image data.
%       CENTERS, detected fibre centers. If CENTERS contains track ids (as
%           returned by fibre_tracking function) those will be shown as 
%           numbers.
%       CLIM, optional color limits.
%   Author: vand@dtu.dk 2018

if ~iscell(centers) % then tracks are given, and need to be converted
    tracks = centers;
    tracks = permute(tracks,[2,3,1]);
    tracks(:,3,:) = repmat((1:size(tracks,1))',[1 1 size(tracks,3)]); % using slice number not z coordinate
    centers = permute(mat2cell(tracks,size(tracks,1),size(tracks,2),ones(size(tracks,3),1)),[3 2 1]);
end

Z = length(imfinfo(filename));
z = round(0.5*(Z+1));
%update_drawing
I = imread(filename,'Index',z);
dim = [size(I),Z];
if nargin<3 || isempty(clim)
    clim = [min(I(:)),max(I(:))];
end

show_numbers = true;

fig = figure('Units','Normalized','Position',[0.1 0.3 0.5 0.6],...
    'KeyPressFcn',@key_press);
imagesc(I,clim)
hold on, colormap gray, axis image ij
plot(centers{z}(:,1),centers{z}(:,2),'r.')
title(['slice ',num2str(z),'/',num2str(Z)])
drawnow

% to capture zoom
LIMITS = [0.5,dim(2)+0.5,0.5,dim(1)+0.5];
zoom_handle = zoom(fig);
pan_handle = pan(fig);
set(zoom_handle,'ActionPostCallback',@adjust_limits,...
    'ActionPreCallback',@force_keep_key_press);
set(pan_handle,'ActionPostCallback',@adjust_limits,...
    'ActionPreCallback',@force_keep_key_press);

% if nargout>1
%     uiwait(fig) % waits with assigning output until a figure is closed
% end

%%%%%%%%%% CALLBACK FUNCTIONS %%%%%%%%%%

    function key_press(~,object)
        % keyboard commands
        key = object.Key;
        switch key
            case 'uparrow'
                z = min(z+1,dim(3));
            case 'downarrow'
                z = max(z-1,1);
            case 'rightarrow'
                z = min(z+10,dim(3));
            case 'leftarrow'
                z = max(z-10,1);
            case 'pageup'
                z = min(z+50,dim(3));
            case 'pagedown'
                z = max(z-50,1);
            case 'n'
                show_numbers = ~show_numbers;
        end
        update_drawing
    end

%%%%%%%%%% HELPING FUNCTIONS %%%%%%%%%%

    function update_drawing
        I = imread(filename,'Index',z);
        cla, imagesc(I,clim), hold on
        if ~show_numbers || size(centers{z},2)<3 || (LIMITS(2)-LIMITS(1))*(LIMITS(4)-LIMITS(3))>300^2
            plot(centers{z}(:,1),centers{z}(:,2),'r.')
        else
            present = (centers{z}(:,1)<LIMITS(2))&(centers{z}(:,1)>LIMITS(1))&...
                (centers{z}(:,2)<LIMITS(4))&(centers{z}(:,2)>LIMITS(3));
            text(centers{z}(present,1),centers{z}(present,2),num2cell(centers{z}(present,3)'),...
                'HorizontalAlignment','center','VerticalAlignment','middle',...
                'Color','r')
        end
        title(['slice ',num2str(z),'/',num2str(Z)])
        drawnow
    end

    function adjust_limits(~,~)
        % response to zooming and panning
        LIMITS([1,2]) = get(gca,'XLim');
        LIMITS([3,4]) = get(gca,'YLim');
        if show_numbers
            update_drawing
        end
    end

    function force_keep_key_press(~,~)
        % a hack to maintain my key_press while in zoom and pan mode
        % http://undocumentedmatlab.com/blog/enabling-user-callbacks-during-zoom-pan
        hManager = uigetmodemanager(fig);
        [hManager.WindowListenerHandles.Enabled] = deal(false);
        set(fig, 'KeyPressFcn', @key_press);
    end

end
