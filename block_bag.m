classdef block_bag
    %% Properties private & public
    properties(SetAccess=private,Hidden=1)
        name = 'Data bags for MIL'
        nameAbb = 'bagMIL'
    end
    
    properties
        Scheme;   % 1.random bag size, 2.uniform bag size
        GenParam; % nPosBag,nNegBag,Random,{PosPortion,BagSize}
        Bag;      % X,Y,y,BagName
    end
    
    %% Methods
    methods
        function self = block_bag(vargin,data)
            self = prtUtilAssignStringValuePairs(self,vargin{:});
            % Check scheme is specified or give error msg
            if isempty(self.Scheme)
                error('Error! Scheme is not specified. Specify Scheme: 1 for Random positive ratio, 2 for Uniform positive ratio');
            end
            % Default value for GenParam if not specified
            if isempty(self.GenParam)
                self.GenParam.nPosBag = 150;
                self.GenParam.nNegBag = 150;
                self.GenParam.Random = 0;
                fprintf('Bags with %d positive bags,%d negative bags,and randomness %d are generated\n',...
                    self.GenParam.nPosBag,self.GenParam.nNegBag,self.GenParam.Random);
                switch self.Scheme
                    case 1
                    
                    case 2
                    self.GenParam.BagSize = 50;
                    self.GenParam.PosPortion = 50;
                    fprintf('Postive portion %d/50\n',...
                        self.GenParam.PosPortion);
                    case 3
                        
                end
            end
            % Generate bags
            if isempty(self.Bag)
                self = self.GenBag(data);
            end
        end
        
        function self = GenBag(self,datablock)
            
            
            % Initialise properties
            self.Bag.X = cell(self.GenParam.nPosBag+self.GenParam.nNegBag,1);
            self.Bag.Y = cell(self.GenParam.nPosBag+self.GenParam.nNegBag,1);
            self.Bag.y = cell(self.GenParam.nPosBag+self.GenParam.nNegBag,1);
            
            nPos = sum(datablock.Data.Y==1);
            nNeg = sum(datablock.Data.Y==-1);
            PosInd = find(datablock.Data.Y==1);
            NegInd = find(datablock.Data.Y==-1);
            if self.GenParam.Random
                PosInd = PosInd(randperm(nPos));
                NegInd = NegInd(randperm(nNeg));
            end
            switch self.Scheme
                case 1
                    % Ground check
                    assert(nPos>=self.GenParam.nPosBag,...
                        'Error! Scheme 1 requires number of positive bags smaller than number of positive instances in data block');
                    assert(nNeg>=self.GenParam.nNegBag,...
                        'Error! Scheme 1 requires number of negative bags smaller than number of negative instances in data block');
                    posIns = randi(floor(nPos/self.GenParam.nPosBag),self.GenParam.nPosBag,1);
                    while sum(posIns)~=nPos
                        ind_add_to = randi(length(posIns));
                        posIns(ind_add_to) = posIns(ind_add_to)+randi(nPos-sum(posIns));
                    end
                    negIns = [zeros(self.GenParam.nPosBag,1);ones(self.GenParam.nNegBag,1)]...
                        +randi(ceil(nNeg/(self.GenParam.nPosBag+self.GenParam.nNegBag))...
                        ,(self.GenParam.nPosBag+self.GenParam.nNegBag),1)-1;
                    while sum(negIns)~=nNeg
                        ind_add_to = randi(length(negIns));
                        negIns(ind_add_to) = negIns(ind_add_to)+randi(nNeg-sum(negIns));
                    end
                    PosNegIns = [[posIns,negIns(1:self.GenParam.nPosBag)];...
                        [zeros(self.GenParam.nNegBag,1),negIns(self.GenParam.nPosBag+1:end)]];
                case 2
                    % Ground check
                    assert(self.GenParam.PosPortion*self.GenParam.nPosBag<=nPos,...
                        'Error! Scheme 2 requires number of positive instance in positive bags smaller than total positive instances in data block');
                    assert(self.GenParam.BagSize*(length(self.Bag.X))-...
                        self.GenParam.PosPortion*self.GenParam.nPosBag<=nNeg,...
                        'Error! Scheme 2 requires more negative instances in data block or less positive portion in positive bags');
                    PosNegIns = [ones(self.GenParam.nPosBag,1)*[self.GenParam.PosPortion,...
                        self.GenParam.BagSize-self.GenParam.PosPortion];...
                        [ones(self.GenParam.nNegBag,1)*[0,self.GenParam.BagSize]]];
            end
            for nBag = 1:length(self.Bag.X)
                self.Bag.X{nBag,1} = [datablock.Data.X(PosInd(1:PosNegIns(nBag,1)),:);...
                    datablock.Data.X(NegInd(1:PosNegIns(nBag,2)),:)];
                self.Bag.Y{nBag,1} = self.signnum(nBag-self.GenParam.nPosBag)...
                    *ones(sum(PosNegIns(nBag,:)),1);
                self.Bag.y{nBag,1} = [datablock.Data.Y(PosInd(1:PosNegIns(nBag,1)));...
                    datablock.Data.Y(NegInd(1:PosNegIns(nBag,2)))];
                PosInd(1:PosNegIns(nBag,1)) = [];
                NegInd(1:PosNegIns(nBag,2)) = [];
            end
            self.Bag.BagName = datablock.MakeName();
        end
        
        function [] = PlotBag(self)
            Data = cell2mat(self.Bag.X);
            LabelsinBag = cell2mat(self.Bag.Y);
            LabelsinIns = cell2mat(self.Bag.y);
            Labels = sign(LabelsinBag+LabelsinIns);
            ds = prtDataSetClass(Data,Labels);
            plot(ds);
%             legend('Neg in Neg Bag','Neg in Pos Bag','Pos in Pos Bag');
        end
        
        function NameStr = MakeName(self)
            name_str{1} = sprintf('scheme%d',self.Scheme);
            name_str{2} = sprintf('nposbag%d',self.GenParam.nPosBag);
            name_str{3} = sprintf('nnegbag%d',self.GenParam.nNegBag);
            name_str{4} = sprintf('random%d',self.GenParam.Random);
            switch self.Scheme
                case 1
                case 2
                    name_str{5} = sprintf('bagsize%d',self.GenParam.BagSize);
                    name_str{6} = sprintf('posportion%d',self.GenParam.PosPortion);
            end
             NameStr = strjoin(string(name_str),'');
        end
        
        function sign = signnum(self,num)
            if num<0
                sign = 1;
            else
                sign = -1;
            end
        end
    end
        
    % Static methods
    methods(Static)
        function saveProperties(block,savedir,filename)
            if ~exist('filename','var')
                filename = block.MakeName();
            end
            savedir = strcat(savedir,block.Bag.BagName,'/');
            if exist(savedir,'dir')==0
                mkdir(char(savedir));
            end
            savedir = strcat(savedir,filename,'.mat');
            if exist(savedir,'file')==0
                save(savedir,'block');
            else
                fprintf(strcat('File already exists:\n ',savedir,'\n'));
            end
        end

        function self = loadProperties(savedir)
            bag = load(savedir);
            self = bag.block;
        end
    end
end