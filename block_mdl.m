classdef block_mdl
    %% Properties private & public
    properties(SetAccess=private,Hidden=1)
        name = 'Model for MIL'
        nameAbb = 'mdlMIL'
    end
    
    properties
        Scheme;   % 1.miSVM 2.adjust miSVM
        GenParam; % Random,Percent,MaxIter,PicSaveDir,{KernelSize,Solver}{Threshold,Step}
                  % ClassType(1.Instance,2.Bag)
        Result;   % Mdl,miAUC,AUC,Converge,Scores,Acc_Train,Acc_Test,Name,BagProperties
        SaveDir = '/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/mdl/'
    end
    
    %% Methods
    methods
        function self = block_mdl(vargin)
            self = prtUtilAssignStringValuePairs(self,vargin{:});
            % Default scheme
            if isempty(self.Scheme)
                self.Scheme = 1;
                disp('Model Default: SVM');
            end
            % Default GenParam
            if isempty(self.GenParam)
                self.GenParam.Random = 0;
                self.GenParam.Percent = 1;
                self.GenParam.MaxIter = 100;
                self.GenParam.KernelSize = 1;
                self.GenParam.Solver = 'SMO';
                self.GenParam.SaveMdl = 0;
                switch self.Scheme
                    case 1
                        
                    case 2
                        self.GenParam.Threshold = 0.5;
                        self.GenParam.Step = 0.1;
                end
            end
        end
        
        function self = run(self,bagblock)
            % Initialise Result
            self.Result.Mdl = cell(self.GenParam.MaxIter,1);
            self.Result.Scores = cell(self.GenParam.MaxIter,1);
            self.Result.AUC = zeros(self.GenParam.MaxIter,1);
            self.Result.Converge = [];
            self.Result.Acc_Train = zeros(self.GenParam.MaxIter,1);
            self.Result.Acc_Test = zeros(self.GenParam.MaxIter,1);
            self.Result.Pd = cell(self.GenParam.MaxIter,1);
            self.Result.Pf = cell(self.GenParam.MaxIter,1);
            
            
            % Prespare training and testing sets
            ds_Train.X = [];
            ds_Train.Y = [];
            ds_Train.y = [];
            ds_Test.X = [];
            ds_Test.Y = [];
            ds_Test.y = [];
            
            % TO DO adapt ntestbag and ntrain bag to distinctive nnegbag
            % and nposbag
            nTestBag = floor(bagblock.GenParam.nPosBag/3*self.GenParam.Percent);
            nTrainBag = nTestBag*2;
            IndPosBag = 1:bagblock.GenParam.nPosBag;
            IndNegBag = 1+bagblock.GenParam.nPosBag:bagblock.GenParam.nPosBag...
                +bagblock.GenParam.nNegBag;
            
            if self.GenParam.Random
                IndPosBag = IndPosBag(randperm(bagblock.GenParam.nPosBag));
                IndNegBag = IndNegBag(randperm(bagblock.GenParam.nNegBag));
            end
            
            IndTrain = [IndPosBag(1:nTrainBag),IndNegBag(1:nTrainBag)];
            IndTest = [IndPosBag(nTrainBag+1:nTrainBag+nTestBag),...
                IndNegBag(nTrainBag+1:nTrainBag+nTestBag)];
            ds_Train.X = cell2mat(bagblock.Bag.X(IndTrain));
            ds_Train.Y = cell2mat(bagblock.Bag.Y(IndTrain));
            ds_Train.y = cell2mat(bagblock.Bag.y(IndTrain));
            ds_Test.X = cell2mat(bagblock.Bag.X(IndTest));
            ds_Test.Y = cell2mat(bagblock.Bag.Y(IndTest));
            ds_Test.y = cell2mat(bagblock.Bag.y(IndTest));
            
            IndPos_Train = find(ds_Train.Y==1);
%             SizeinBag = cellfun(@size,self.Bag.X,'uni',false);
%             SizeinBag = SizeinBag(:,1);
%             nDim = size(ds_Train.X,2);
            
            % MIL iterations
            OutputLabels = ds_Train.Y;
            InputLabels = OutputLabels-1;
            nIter = 0;
            KernelSize = self.GenParam.KernelSize;
            
            while sum(OutputLabels-InputLabels~=0)~=0 && nIter<self.GenParam.MaxIter
                nIter = nIter+1;
                InputLabels = OutputLabels;
                Mdl = fitcsvm(ds_Train.X,InputLabels,...
                    'Solver',self.GenParam.Solver,...
                    'KernelFunction','rbf',...
                    'KernelScale',KernelSize);
                [Labels_Train,Scores_Train] = predict(Mdl,ds_Train.X);
                OutputLabels(IndPos_Train) = Labels_Train(IndPos_Train);
                switch bagblock.Scheme
                    case 1
                        
                    case 2
                       if prod(sum(reshape(Labels_Train(IndPos_Train),bagblock.GenParam.BagSize,[])+1)) ==0
                           IndBag = find(sum(reshape(Labels_Train(IndPos_Train),bagblock.GenParam.BagSize,[])+1)==0);
                           sc = Scores_Train(:,2);
                           sc = sc(IndPos_Train);
                           sc = reshape(sc,bagblock.GenParam.BagSize,[]);
                           for ind = IndBag
                               [~,maxIns] = max(sc(:,ind));
                               OutputLabels(IndPos_Train((ind-1)*bagblock.GenParam.BagSize+maxIns)) = 1;
                           end
                       end
                end
                
                switch self.Scheme
                    case 1
                    case 2
                        KernelSize = max(KernelSize-self.GenParam.Step,...
                            self.GenParam.Threshold);
                end
                
                [Labels_Test,Scores_Test] = predict(Mdl,ds_Test.X);
                % Save properties each iteration
                if self.GenParam.SaveMdl
                    self.Result.Mdl{nIter,1} = Mdl;
                end
                self.Result.Scores{nIter,1} = Scores_Train(:,2);
                [self.Result.Pd{nIter,1},self.Result.Pf{nIter,1},~,self.Result.AUC(nIter)] = perfcurve(ds_Test.y,Scores_Test(:,2),1);
                 self.Result.Converge = [self.Result.Converge,sum(OutputLabels-InputLabels~=0)];
                self.Result.Acc_Train(nIter) = sum(Labels_Train-ds_Train.y==0)/length(Labels_Train);
                self.Result.Acc_Test(nIter) = sum(Labels_Test-ds_Test.y==0)/length(Labels_Test);
            end
            
            self.SaveDir = strcat(self.SaveDir,bagblock.Bag.BagName,'/');
            self.Result.Name = bagblock.MakeName();
            self.Result.BagProperties = bagblock.GenParam;
            self.Result.BagScheme = bagblock.Scheme;
        end
        
        function [] = PlotDecBound(self,PicSaveDir)
            if ~exist('PicSaveDir','var')
                PicSaveDir = self.SaveDir;
            end
            PicSaveDir = strcat(PicSaveDir,self.MakeName(),'decisionboundary/');
            if self.Result.BagScheme ==2
                PicSaveDir = strcat(PicSaveDir,sprintf('posportion%d',...
                    self.Result.BagProperties.PosPortion),'/');
            end
            if exist(PicSaveDir,'dir')==0
                mkdir(char(PicSaveDir));
            end
            nIter = length(self.Result.Converge);
            for niter = 1:nIter
                Mdl = self.Result.Mdl{niter,1};
                ds = prtDataSetClass(Mdl.X,Mdl.Y);
                gamma = 1/(Mdl.KernelParameters.Scale^2);
                classifier = prtClassLibSvm('kernelType',2,'gamma',gamma);
                classified = classifier.train(ds);
                figure(1);
                plot(classified);
                title(sprintf('Decision Boundary at Iteration %d',niter));
                saveas(figure(1),char(strcat(PicSaveDir,sprintf('decisionboundaryIter%d.png',niter))));
            end
        end
        
        function NameStr = MakeName(self)
            str_name{1} = sprintf('Percent%d/',self.GenParam.Percent*100);
            switch self.Scheme
                case 1
                    str_name{2} = sprintf('Sigma%.2f/',self.GenParam.KernelSize);                   
                case 2
                    str_name{2} = sprintf('Threshold%.2f/',self.GenParam.Threshold);
                    str_name{3} = sprintf('Step%.2f/',self.GenParam.Step);
                    str_name{4} = sprintf('Sigma%.2f/',self.GenParam.KernelSize);
            end
            NameStr = strjoin(string(str_name),'');
        end
    end
    
    methods(Static)
        function saveProperties(mdlblock,savedir,filename)
            if isempty(savedir) || ~exist('savedir','var')
                savedir = mdlblock.SaveDir;
            end
            if ~exist('filename','var')
                filename = mdlblock.Result.Name;
            end
            savedir = strcat(savedir,mdlblock.MakeName());
            if exist(savedir,'dir')==0
                mkdir(char(savedir));
            end
            savedir = strcat(savedir,filename,'.mat');
            if exist(savedir,'file')==0
                save(savedir,'mdlblock');
            else
                fprintf(strcat('File already exists:\n ',savedir));
            end
        end
        
        function self = loadProperties(savedir)
            mdl = load(savedir);
            self = mdl.mdlblock;
        end
    end
end