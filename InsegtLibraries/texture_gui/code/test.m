addpath('functions')
%%
fig = figure;
imagesc(rand(200,200));
set(fig,'MenuBar','none','Toolbar','figure');

%%
toolbar = findall(fig,'Type','uitoolbar');
all_tools = allchild(toolbar);
% removing tools
for i=1:numel(all_tools)
    t = get(all_tools(i),'Tag');
    if isempty(strfind(t,'Pan'))&&...
            isempty(strfind(t,'Zoom'))&&...
            isempty(strfind(t,'SaveFigure'))&&...
            isempty(strfind(t,'PrintFigure'))&&...
            isempty(strfind(t,'DataCursor'))
        delete(all_tools(i)) % keeping only Pan, Zoom, Save and Print
    end
end
%% adding a tool
[icon,~,alpha] = imread('linkaxesicon.png');
icon = double(icon)/255;
icon(alpha==0)=NaN;
uitoggletool(toolbar,'CData',icon,...
    'TooltipString','Link Axes','Tag','LinkAxes',...
    'OnCallback',{@link_axes,'xy'},...
    'OffCallback',{@link_axes,'off'});
%% changing the order of tools
all_tools = allchild(toolbar);
set(toolbar,'children',all_tools([2,1,3:end]));