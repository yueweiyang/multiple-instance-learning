clear all;
close all;

ToGenData = 0;
ToGenBag = 0;

% Generate data block
DataScheme = 2;
DataGenParam.nSamples = 30000;
DataGenParam.Noise = .3;
DataGenParam.nClusters = 2;
DataSaveDir = '/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/data/';
if ToGenData
    data = block_data({'Scheme',DataScheme,'GenParam',DataGenParam});
    block_data.saveProperties(data,DataSaveDir);
else
    switch DataScheme
        case 1
            str_data = 'Gaussians';
        case 2
            str_data = 'Spirals';
    end
    str_data = strcat(str_data,sprintf('size%dnclusters%dnoise%.2f.mat',DataGenParam.nSamples,...
        DataGenParam.nClusters,DataGenParam.Noise));
    data = block_data.loadProperties(strcat(DataSaveDir,str_data));
end

% Bag block parameters
BagScheme = 2;
BagGenParam.nPosBag = 150;
BagGenParam.nNegBag = 150;
BagGenParam.Random = 1;
BagGenParam.BagSize = 50;
BagSaveDir = sprintf('/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/bag/data noise %.2f/',DataGenParam.Noise);
if ToGenBag
    for dataset = 1:100
        for posportion = [1,5:5:50]
            str_posportion = sprintf('PosPortion%d',posportion);
            BagGenParam.PosPortion = posportion;
            bag = block_bag({'Scheme',BagScheme,'GenParam',BagGenParam},data);
            bag.Bag.BagName = sprintf('DataSet%d',dataset);
            block_bag.saveProperties(bag,BagSaveDir,str_posportion);
        end
    end
end


% Model block parameters
MdlScheme = 2;
MdlGenParam.Random = 1;
MdlGenParam.Percent = .5;
MdlGenParam.MaxIter = 30;
MdlGenParam.Solver = 'SMO';
MdlSaveDir = sprintf('/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/mdl/noise%.2f/',DataGenParam.Noise);
for dataset = 1:10
    if dataset == 1
        MdlGenParam.SaveMdl = 1;
    else
        MdlGenParam.SaveMdl = 0;
    end
    for posportion = [1,5:5:50]
        str_posportion = sprintf('PosPortion%d',posportion);
        str_bagfile = strcat(sprintf('DataSet%d/',dataset),str_posportion,'.mat');
        bag = block_bag.loadProperties(strcat(BagSaveDir,str_bagfile));
        for sigma = [0.01:0.04:0.2]
            MdlGenParam.KernelSize = sigma;
            for threshold = [0.01:0.05:0.2]
                MdlGenParam.Threshold = threshold;
                for step = [0.02,0.05,0.1]
                    MdlGenParam.Step = step;
                    str_mdlfile = strcat(MdlSaveDir,sprintf(...
                        'DataSet%d/Percent%d/Threshold%.2f/Step%.2f/Sigma%.2f/PosPortion%d',...
                        dataset,MdlGenParam.Percent*100,threshold,step,sigma,posportion),'.mat');
                    if ~exist(str_mdlfile,'file')
                        mdl = block_mdl({'Scheme',MdlScheme,'GenParam',MdlGenParam});
                        mdl = mdl.run(bag);
                        block_mdl.saveProperties(mdl,strcat(MdlSaveDir,sprintf('DataSet%d/',dataset)),str_posportion);
                    else
                        fprintf(strcat('File exist \n',str_mdlfile));
                    end
                end
            end
        end
    end
end
            