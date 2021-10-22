function segmentation_correction_gui(image,segmentation,nr_labels,labeling)
%IMAGE_TEXTURE_GUI   Interactive segmentation correction

%%%%%%%%%% DEFAULTS %%%%%%%%%%

if nargin<1 % default example image
    %image = imread('bag.png');
    image = imread('football.jpg');
end
[r,c,~] = size(image);
if nargin<2 || isempty(segmentation)% default segmentation
    segmentation = zeros(r,c,'uint8');
else
    segmentation = uint8(segmentation);
end
if nargin<3 || isempty(nr_labels) % defalult number of labels
    nr_labels = double(max(2,max(segmentation(:))));
end
if nargin<4 || isempty(labeling) % default, unlabeled initial correction
    % TODO allow corrections to be inferred from labelings
    DRAWING = zeros(r,c,'uint8');
else
    DRAWING = uint8(labeling);
end

%%%%%%%%%% SETTINGS %%%%%%%%%%

% labels
LABEL = 1; % initial label is 1

% thickness
thickness_options = [1 2 3 4 5 10 20 30 40 50 100 -1]; % last option (value -1) is 'fill'
thickness_options_string = num2cell(thickness_options);
thickness_options_string{end} = 'fill';
THICKNESS_INDEX = 5; % initial pencil thickness is the fifth option
RADIUS = thickness2radius(thickness_options(THICKNESS_INDEX)); % pencil radius

% show
show_options(1:2) = {'segmentation','overlay'};
SHOW_INDEX = 1;

% colormap
colormap_options = {@(x)[0.5,0.5,0.5;cool(x)], @(x)[0.5,0.5,0.5;spring(x)],...
    @(x)[0.5,0.5,0.5;parula(x)]}; % gray color for unlabeled
COLORMAP_INDEX = 1; % current colormap
COLORS = colormap_options{COLORMAP_INDEX}(nr_labels); % visualization colors for label overlay
% other settings
color_weight = 0.4; % color weight for label overlay
nr_circle_pts = 16; % number of points defining a circular pencil

%%%%%%%%%% INITIALIZATION AND LAYOUT %%%%%%%%%%
[image,gray] = normalize_image(image); % impose 0-to-1 rgb
CORRECTED = uint8(segmentation);
CORRECTED(DRAWING~=0) = DRAWING(DRAWING~=0); % if initial corrections given
DRAWING_OVERLAY = image; % image overlaid labeling
CORRECTED_OVERLAY = ind2rgb(CORRECTED,COLORS);

fmar = [0.2 0.2]; % discance from screen edge to figure (x and y)
amar = [0.05 0.05]; % margin around axes, relative to figure
my = 0.8:0.05:0.9; % menu items y position
mh = 0.03; % menu items height
cw = (0.4-0.3)/(nr_labels+1); % colorcube width
cx = 0.3:cw:0.4; % colorcubes x position

fig = figure('Units','Normalized','Position',[fmar,1-2*fmar],...
    'Pointer','watch','KeyPressFcn',@key_press,'InvertHardCopy', 'off',...
    'Name','Segmentation correction GUI');
clean_toolbar

labeling_axes = axes('Units','Normalized','Position',[0,0,0.5,0.8]+[amar,-2*amar]);
imagesc(DRAWING_OVERLAY), axis image off, hold on
segmentation_axes = axes('Units','Normalized','Position',[0.5,0,0.5,0.8]+[amar,-2*amar]);
imagesc(CORRECTED_OVERLAY,[0,nr_labels]), axis image off, hold on

uicontrol('String','Label [L] : ','Style','text','HorizontalAlignment','right',...
    'Units','Normalized','Position',[0,my(3),0.25,mh]);
labels_text = uicontrol('String',num2str(LABEL),...
    'BackgroundColor',COLORS(LABEL+1,:),...
    'Style','text','HorizontalAlignment','left',...
    'Units','Normalized','Position',[0.25,my(3),0.03,mh]);
label_pointer = cell(nr_labels+1,1);
label_colorcubes = cell(nr_labels+1,1);
label_char = repmat(' ',[1,nr_labels+1]);
label_char(LABEL+1)='|';
for k = 1:nr_labels+1
    label_colorcubes{k} = uicontrol('String',' ','Style','text',...
        'BackgroundColor',COLORS(k,:),...
        'Units','Normalized','Position',[cx(k),my(3),cw,mh/2]);
    label_pointer{k} = uicontrol('String',label_char(k),'Style','text',...
        'HorizontalAlignment','center',...
        'Units','Normalized','Position',[cx(k),my(3)+mh/2,cw,mh/2]);
end

uicontrol('String','Thickness [T] : ','Style','text','HorizontalAlignment','right',...
    'Units','Normalized','Position',[0,my(2),0.25,mh]);
thickness_text = uicontrol('String',thickness_options_string(THICKNESS_INDEX),...
    'Style','text','HorizontalAlignment','left',...
    'Units','Normalized','Position',[0.25,my(2),0.25,mh]);

uicontrol('String','Show [W] :','Style','text','HorizontalAlignment','right',...
    'Units','Normalized','Position',[0,my(1),0.25,mh]);
show_text = uicontrol('String',show_options{SHOW_INDEX},...
    'Style','text','HorizontalAlignment','left',...
    'Units','Normalized','Position',[0.25,my(1),0.25,mh]);

drawnow % pointer shows busy system

LIMITS = [1,c-0.5,1,r+0.5]; % to capture zoom
zoom_handle = zoom(fig);
pan_handle = pan(fig);
set(zoom_handle,'ActionPostCallback',@adjust_limits,...
    'ActionPreCallback',@force_keep_key_press);
set(pan_handle,'ActionPostCallback',@adjust_limits,...
    'ActionPreCallback',@force_keep_key_press);

compute_overlays

% ready to draw
set(fig,'Pointer','arrow','WindowButtonDownFcn',@start_draw,...
    'WindowButtonMotionFcn',@pointer_motion);
XO = []; % current drawing point
uiwait % waits with assigning output until a figure is closed

%%%%%%%%%% CALLBACK FUNCTIONS %%%%%%%%%%
    function key_press(~,object)
        % keyboard commands
        key = object.Key;
        numkey = str2double(key);
        if ~isempty(numkey) && numkey<=nr_labels;
            label_char(LABEL+1)=' ';
            LABEL = numkey;
            set(labels_text,'String',num2str(LABEL),...
                'BackgroundColor',COLORS(LABEL+1,:));
            label_char(LABEL+1)='|';
            for kk = 1:nr_labels+1
                set(label_pointer{kk},'String',label_char(kk));
            end
        else
            switch key
                case 'l'
                    label_char(LABEL+1)=' ';
                    LABEL = move_once(LABEL+1,nr_labels+1,...
                        any(strcmp(object.Modifier,'shift')))-1;
                    set(labels_text,'String',num2str(LABEL),...
                        'BackgroundColor',COLORS(LABEL+1,:));
                    label_char(LABEL+1)='|';
                    for kk = 1:nr_labels+1
                        set(label_pointer{kk},'String',label_char(kk));
                    end
                case 'uparrow'
                    THICKNESS_INDEX = move_once(THICKNESS_INDEX,...
                        numel(thickness_options),false);
                    RADIUS = thickness2radius(thickness_options(THICKNESS_INDEX));
                    set(thickness_text,'String',...
                        thickness_options_string(THICKNESS_INDEX));
                    show_overlays(get_pixel);
                case 'downarrow'
                    THICKNESS_INDEX = move_once(THICKNESS_INDEX,...
                        numel(thickness_options),true);
                    RADIUS = thickness2radius(thickness_options(THICKNESS_INDEX));
                    set(thickness_text,'String',...
                        thickness_options_string(THICKNESS_INDEX));
                    show_overlays(get_pixel);
                case 't'
                    THICKNESS_INDEX = move_once(THICKNESS_INDEX,...
                        numel(thickness_options),any(strcmp(object.Modifier,'shift')));
                    RADIUS = thickness2radius(thickness_options(THICKNESS_INDEX));
                    set(thickness_text,'String',...
                        thickness_options_string(THICKNESS_INDEX));
                    show_overlays(get_pixel);
                case 'w'
                    SHOW_INDEX = move_once(SHOW_INDEX,numel(show_options),...
                        any(strcmp(object.Modifier,'shift')));
                    set(show_text,'String',show_options{SHOW_INDEX})
                    compute_overlays
                    show_overlays(get_pixel);
                case 'c'
                    COLORMAP_INDEX = move_once(COLORMAP_INDEX,...
                        length(colormap_options),...
                        any(strcmp(object.Modifier,'shift')));
                    COLORS = colormap_options{COLORMAP_INDEX}(nr_labels);
                    set(labels_text,'BackgroundColor',COLORS(LABEL+1,:));
                    for kk = 1:nr_labels+1
                        set(label_colorcubes{kk},'BackgroundColor',COLORS(kk,:))
                    end
                    compute_overlays
                    show_overlays(get_pixel);
                case 's'
                    save_project
                case 'e'
                    export_DC
                case 'q'
                    close(fig)
            end
        end
    end

    function adjust_limits(~,~)
        % response to zooming and panning
        LIMITS([1,2]) = get(labeling_axes,'XLim');
        LIMITS([3,4]) = get(labeling_axes,'YLim');
    end

    function force_keep_key_press(~,~)
        % a hack to maintain my key_press while in zoom and pan mode
        % http://undocumentedmatlab.com/blog/enabling-user-callbacks-during-zoom-pan
        hManager = uigetmodemanager(fig);
        [hManager.WindowListenerHandles.Enabled] = deal(false);
        set(fig, 'KeyPressFcn', @key_press);
    end

    function start_draw(~,~)
        % click in the image
        x = get_pixel;
        if is_inside(x)
            if RADIUS>0 % thickness>0
                XO = x;
                M = disc(XO,RADIUS,nr_circle_pts,[r,c]);
                set(fig,'WindowButtonMotionFcn',@drag_and_draw,...
                    'WindowButtonUpFcn',@end_draw)
            else % fill
                M = fill(x);
            end
            update(M);
        end
    end

    function drag_and_draw(~,~)
        % drag after clicking in the image
        x = get_pixel;
        M = stadium(XO,x,RADIUS,nr_circle_pts,[r,c]);
        update(M);
        XO = x;
    end

    function end_draw(~,~)
        % release click after clicking in the image
        M = stadium(XO,get_pixel,RADIUS,nr_circle_pts,[r,c]);
        update(M);
        XO = [];
        set(fig,'WindowButtonMotionFcn',@pointer_motion,...
            'WindowButtonUpFcn','')
    end

    function pointer_motion(~,~)
        % move around without clicking
        if strcmp(zoom_handle.Enable,'off') && ...
                strcmp(pan_handle.Enable,'off') % not zooming or panning
            x = get_pixel;
            if is_inside(x)
                set(fig,'Pointer','crosshair')
            else
                set(fig,'Pointer','arrow')
            end
            show_overlays(x);
        end
    end

%%%%%%%%%% HELPING FUNCTIONS %%%%%%%%%%

    function save_project
        % save mat file with user input and setting
        % save images as separate files
        [file,path] = uiputfile('correction.mat','Save project as');
        if ~isequal(file,0) && ~isequal(path,0)
            matfile = fullfile(path,file);
            roothname = matfile(1:find(matfile=='.',1,'last')-1);
            current_settings.show = show_options{SHOW_INDEX};
            save(matfile,'DRAWINGS','CORRECTED',...
                'dictopt','nr_labels','current_settings')
            imwrite(DRAWING_OVERLAY,[roothname,'_drawing.png'])
            imwrite(CORRECTED_OVERLAY,[roothname,'_corrected.png'])
        end
    end

    function export_DC
        button = questdlg({'Exporting variables to the base workspace',...
            'might overwrite existing variables'},...
            'Exporting variables','OK','Cancel','OK');
        if strcmp(button,'OK')
            assignin('base','gui_D',DRAWING)
            assignin('base','gui_C',CORRECTED)
        end
        
    end

    function a = is_inside(x)
        % check if x is inside image limits
        a = inpolygon(x(1),x(2),LIMITS([1,2,2,1]),LIMITS([3,3,4,4]));
    end

    function p = get_pixel
        % get cursor position
        p = get(labeling_axes,'CurrentPoint');
        p = round(p(1,[1,2]));
    end

    function show_overlays(x)
        % overlay a circular region where the pointer and show
        shown_left = DRAWING_OVERLAY;
        shown_right = CORRECTED_OVERLAY;
        if RADIUS>0 % thickness>0
            P = repmat(disc(x,RADIUS,nr_circle_pts,[r,c]),[1,1,3]);
            shown_left(P(:)) = 0.5+0.5*DRAWING_OVERLAY(P(:));
            shown_right(P(:)) = 0.5+0.5*CORRECTED_OVERLAY(P(:));
        end
        % we have to imagesc(shown) to remove overlay if needed
        axes(labeling_axes), cla, imagesc(shown_left)
        axes(segmentation_axes), cla, imagesc(shown_right)
    end

    function update(M)
        % change the state of the segmentation by updating LABELINGS with a
        % mask M, and updating PROBABILITIES
        DRAWING(M(:)) = LABEL;
        if LABEL>0
            CORRECTED(M(:)) = LABEL;
        else
            CORRECTED(M(:)) = segmentation(M(:));
        end
        compute_overlays % computing overlay images
        show_overlays(get_pixel); % showing overlay and pointer
    end

    function compute_overlays
        % computes overlays but not pointer overalay
        uncorrected = repmat(DRAWING==0,[1 1 3]);
        DRAWING_OVERLAY = uncorrected.*image + ~uncorrected.*...
            (color_weight.*ind2rgb(DRAWING,COLORS) + (1-color_weight).*gray);
        switch show_options{SHOW_INDEX}
            case 'segmentation'
                CORRECTED_OVERLAY = ind2rgb(CORRECTED,COLORS);
            case 'overlay'
                CORRECTED_OVERLAY = color_weight*...
                    ind2rgb(CORRECTED,COLORS) + (1-color_weight)*gray;
        end
    end

% TODO disc shold be saved as a list of index shifts with respect to
% the central pixel, and change only when thickness changes
    function M = disc(x,r,N,dim)
        % disc shaped mask in the image
        angles = (0:2*pi/N:2*pi*(1-1/N));
        X = x(1)+r*cos(angles);
        Y = x(2)+r*sin(angles);
        M = poly2mask(X,Y,dim(1),dim(2));
    end

    function M = stadium(x1,x2,r,N,dim)
        % stadium shaped mask in the image
        angles = (0:2*pi/N:pi)-atan2(x1(1)-x2(1),x1(2)-x2(2));
        X = [x1(1)+r*cos(angles), x2(1)+r*cos(angles+pi)];
        Y = [x1(2)+r*sin(angles), x2(2)+r*sin(angles+pi)];
        M = poly2mask(X,Y,dim(1),dim(2));
    end

    function M = fill(x)
        M = bwselect(DRAWING==DRAWING(x(2),x(1)),x(1),x(2),4);
    end

    function [I,G] = normalize_image(I)
        % initialization: normalize image
        if isa(I,'uint8')
            I = double(I)/255;
        end
        if isa(I,'uint16')
            I = double(I)/65535;
        end
        if size(I,3)==3 % rgb image
            G = repmat(rgb2gray(I),[1 1 3]);
        else % assuming grayscale image
            I = repmat(I,[1,1,3]);
            G = I;
        end
    end

    function n = move_once(n,total,reverse)
        % moves option index once, respecting total number of options
        if ~reverse
            n = mod(n,total)+1;
        else
            n = mod(n+total-2,total)+1;
        end
    end

    function r = thickness2radius(t)
        r = t/2+0.4;
    end

    function clean_toolbar
        set(fig,'MenuBar','none','Toolbar','figure');
        all_tools = allchild(findall(fig,'Type','uitoolbar'));
        %my_tool = uipushtool(toolbar,'CData',rand(16,16,3),'Separator','on',...
        %            'TooltipString','my tool','Tag','my tool');
        for i=1:numel(all_tools)
            if isempty(strfind(all_tools(i).Tag,'Pan'))&&...
                    isempty(strfind(all_tools(i).Tag,'Zoom'))&&...
                    isempty(strfind(all_tools(i).Tag,'SaveFigure'))&&...
                    isempty(strfind(all_tools(i).Tag,'PrintFigure'))&&...
                    isempty(strfind(all_tools(i).Tag,'DataCursor'))
                delete(all_tools(i)) % keeping only Pan, Zoom, Save and Print
            end
        end
    end

end
