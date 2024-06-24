clc;clear all;close all;
%  , 'soda'
datasets = {'cityscapes'};
path = 'E:\Work\Deep_Learning\Thermal_Segmentation\Dataset/'
labels_path = {'Cityscapes_thermal\CITYSCAPE_5000/', 'InfraredSemanticLabel-20210430T150555Z-001\SODA/','Cityscapes\gtFine/' };

% numWorker = 6; % Number of matlab workers for parallel computing
% delete(gcp('nocreate'));
% parpool('local', numWorker);
for idxSet = 1:length(datasets)
    if strcmp(datasets{idxSet}, 'cityscapes_thermal')
        setList = {'train'};
        data_path = labels_path{1}
    else
        setList = {'train', 'val', 'test'};
        data_path = labels_path{3}
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
            edgeMapBin = seg2edge(mask, 1, []', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
            imwrite(edgeMapBin, [save_path '/' fileList(idxFile).name])
        end
    end
end
