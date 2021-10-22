function [rect,im_rect] = selectFoV_VolumetricTiff(im)
%SELECTFOV is a user-friendly function to select a region of interest from a 2D image.
%   The user can choose to select interactively the region of interest (RoI) 
%   interactively with imcrop, or specify the coordinates of the RoI.
% by Monica Jane Emerson, March 2019, MIT License

%USER INPUT: SLICE_PATH, The location of the slice to crop the RoI from.
% Other user input given through interaction with windows.

autoManual = questdlg('How would you like to select the region of interest (RoI)?','Select RoI','Interactively','I know the coordinates','I know the coordinates');

if strcmp(autoManual,'Interactively')
    [~,rect] = imcrop(im);
    rect = round(rect);
    rect(3:4) = rect(1:2) + rect(3:4) - [1,1];
else
    xmin = ['Lower limit for the x value [1, ',num2str(size(im,2)),']:'];
    xmax = ['Upper limit for the x value [1, ',num2str(size(im,2)),']:'];
    ymin = ['Lower limit for the y value [1, ',num2str(size(im,1)),']:'];
    ymax = ['Upper limit for the y value [1, ',num2str(size(im,1)),']:'];
    prompt = {xmin,ymin,xmax,ymax};
    dimRoI = inputdlg(prompt,'RoI in the cross-sectional direction', [1,60], {'1','1',num2str(size(im,2)),num2str(size(im,1))});
    xMin = str2double(dimRoI{1});
    xMax = str2double(dimRoI{3});
    yMin = str2double(dimRoI{2});
    yMax = str2double(dimRoI{4});
    widthX = xMax-xMin+1;
    widthY = yMax-yMin+1;
    while widthX<10 || widthY<10 || xMin<1 || yMin<1 || xMax>size(im,2) || yMax>size(im,1)
        if widthX<0
            message= ...
            sprintf('Wrong inputs, max<min for x, given as: [%.d,%.d], please enter again',...
            xMin,xMax);
        elseif widthY<0
            message= ...
            sprintf('Wrong inputs, max<min for y, given as: [%.d,%.d], please enter again',...
            yMin,yMax);
        elseif widthX<10
            message= ...
            sprintf('Wrong inputs, width<10 for x, given as: [%.d,%.d], please enter again',...
            xMin,xMax);
        elseif widthY<10
            message= ...
            sprintf('Wrong inputs, width<10 for y, given as: [%.d,%.d], please enter again',...
            yMin,yMax);
        elseif xMin<1 || xMax>size(im,2)
            message= ...
            sprintf('Wrong inputs, x given as: [%.d,%.d] outside of data range [%.d,%.d], please enter again',...
            xMin,xMax,1,num2str(size(im,2)));
        elseif yMin<1 || yMax>size(im,1)
            message= ...
            sprintf('Wrong inputs, y given as: [%.d,%.d] outside of data range [%.d,%.d], please enter again',...
            yMin,yMax,1,num2str(size(im,2)));
        end
        
        dimRoI = inputdlg(prompt,message, [1,130], {'1','1',num2str(size(im,2)),num2str(size(im,1))});
        xMin = str2double(dimRoI{1});
        xMax = str2double(dimRoI{3});
        yMin = str2double(dimRoI{2});
        yMax = str2double(dimRoI{4});
        widthX = xMax-xMin+1;
        widthY = yMax-yMin+1;
    end
    
    rect = str2double(dimRoI);
end
    im_rect = im2double(im(rect(2):rect(4),rect(1):rect(3)));

end

