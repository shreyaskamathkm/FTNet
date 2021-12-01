clc;clear all;close all;
%  , 'soda'
datasets = {'cityscapes','soda','scutseg','mfn'};
path = 'D:\processed_dataset/';
labels_path = {'CITYSCAPE_5000/', 'SODA/', 'SCUTSEG/','MFN/',  };
radius = [2,1,1,1];
% numWorker = 6; % Number of matlab workers for parallel computing
% delete(gcp('nocreate'));
% parpool('local', numWorker);
for idxSet = 1:length(datasets)
    if strcmp(datasets{idxSet}, 'cityscapes')
        setList = {'train'};
        data_path = labels_path{1};
        r = radius(1);
    elseif strcmp(datasets{idxSet}, 'soda')
        setList = {'train', 'val', 'test'};
        data_path = labels_path{2};
        r = radius(2);
    elseif strcmp(datasets{idxSet}, 'scutseg')
        setList = {'train', 'test'};
        data_path = labels_path{3};
        r = radius(3);
    elseif strcmp(datasets{idxSet}, 'mfn')
        setList = {'train', 'val', 'test'};
        data_path = labels_path{4};
        r = radius(4);
    end
    
    for set_idx = 1:length(setList)
        save_path = [path data_path 'edges/' setList{set_idx} ];
        if(exist(save_path, 'file')==0)
            mkdir(save_path);
        end
        fileList = dir([path data_path 'mask/' setList{set_idx} ]);
        fileList=fileList(~ismember({fileList.name},{'.','..'}));
        for idxFile = 1 :length(fileList)
            mask = imread([fileList(idxFile).folder '/' fileList(idxFile).name]);
            %SODA=1; Cityscapes = 2;
            edgeMapBin = seg2edge(mask, r, []', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
            imwrite(edgeMapBin, [save_path '/' fileList(idxFile).name])
        end
    end
end




