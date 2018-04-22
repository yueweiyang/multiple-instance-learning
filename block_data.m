classdef block_data
    %% Properties private & public
    properties(SetAccess=private,Hidden=1)
        name = "Data generation for MIL"
        nameAbb = "dataGenMIL"
    end
    
    properties
        Scheme;   %Scheme:1.Gaussian 2.Spiral 3.TBD
        GenParam; %nSamples, Noise, {nClusters,ClusterLabels,Sigma,Mu}
        Data;     %X, Y, Name
    end
    
    %% Methods
    methods
        function self = block_data(vargin)
            self = prtUtilAssignStringValuePairs(self,vargin{:});
            % Check scheme is specified or give error msg
            if isempty(self.Scheme)
                error('Error! Scheme is not specified. Specify Scheme: 1 for Gaussians, 2 for Spirals');
            end
            % Default GenParam if not specified
            if isempty(self.GenParam)
                self.GenParam.nSamples = 1000;
                self.GenParam.Noise = 0;
                self.GenParam.nClusters = 2;
                self.GenParam.ClusterLabels = [1,-1];
                switch self.Scheme
                    case 1       
                        self.GenParam.Sigma = [eye(2);eye(2)];
                        self.GenParam.Mu = [0,0;0,5];
                        disp('Default: Two standard gaussian distributions centered at [0,0] and [0,5] are generated');
                    case 2
                    case 3
                end
            end
            % Generate data
            if isempty(self.Data)
                self = self.genData(); %generate data if not given
            end
        end
        
        function self = genData(self)
            % Default value for GenParam if not specified
            
            
            % Initialise Data
            self.Data.X = [];
            self.Data.Y = [];
            
            switch self.Scheme
                % Scheme 1 Gausisan Distributions
                case 1
                    % Ground check 
                    nDim = size(self.GenParam.Mu,2);
                    assert(size(self.GenParam.Sigma,2)==nDim,...
                        'Error: Dimension mismatch in Sigma and Mu');
                    assert(self.GenParam.nClusters==size(self.GenParam.Mu,1),...
                        'Error: Dimension mismatch in nClusters and Mu');
                    assert(self.GenParam.nClusters==size(self.GenParam.Sigma,1)/nDim,...
                        'Error: Dimension mismatch in nClusters and Sigma');
                    assert(self.GenParam.nClusters==length(self.GenParam.ClusterLabels),...
                        'Error: Dimension mismatch in nClusters and ClusterLabels');
                    
                    % Generate Gaussians based on Sigma, Mu, Noise
                    for nclusters = 1:self.GenParam.nClusters
                        mu = self.GenParam.Mu(nclusters,:);
                        sigma = self.GenParam.Sigma((nclusters-1)*nDim+1:nclusters*nDim,:);
                        noise = self.GenParam.Noise.*(randn(size(mu)));
                        rv = prtRvMvn('mu',mu+noise,'sigma',sigma);
                        self.Data.X = [self.Data.X;draw(rv,self.GenParam.nSamples)];
                        self.Data.Y = [self.Data.Y;ones(self.GenParam.nSamples,1)*...
                        self.GenParam.ClusterLabels(nclusters)];
                    end 
                    self.Data.Name = 'Gaussians';
                % Scheme 2 Spiral Distributions
                case 2
                    % Ground check
                    assert(self.GenParam.nClusters==2,...
                        'Error: Scheme 2 can only generate 2 distributions');
                    
                    % Generate Spirals based on Noise
                    ds = prtDataGenSpiral(self.GenParam.nSamples);
                    angle = 30; %Rotation angle to alter original Spirals
                    ds.data(ds.targets==1,:) = ds.data(ds.targets==1,:)...
                        *[cos(angle),-sin(angle);sin(angle),cos(angle)]+[0,0.5];
                    self.Data.X = ds.data+self.GenParam.Noise.*(randn(size(ds.data)));
                    self.Data.Y = ds.targets;
                    self.Data.Y(self.Data.Y==1) = -1;
                    self.Data.Y(self.Data.Y==0) = 1;
                    self.Data.Name = 'Spirals';
                % Scheme 3 TBD
                case 3
            end
        end
        
        function [] = PlotData(self)
            ds = prtDataSetClass(self.Data.X,self.Data.Y);
            plot(ds);
        end
        
        function NameStr = MakeName(self)
            str_name{1} = self.Data.Name;
            str_name{2} = sprintf('size%d',self.GenParam.nSamples);
            str_name{3} = sprintf('nclusters%d',self.GenParam.nClusters);
            str_name{4} = sprintf('noise%.2f',self.GenParam.Noise);
            switch self.Scheme
                case 1
                    nDim = size(self.GenParam.Mu,2);
                    if nDim<=3
                        mx_Sigma = self.GenParam.Sigma';
                        mx_Mu = self.GenParam.Mu';
                        str_name{5} = strcat('labels',strjoin(string(self.GenParam.ClusterLabels),''));
                        str_name{6} = strcat('sigma',strjoin(string(mx_Sigma(:)),''));
                        str_name{7} = strcat('mu',strjoin(string(mx_Mu(:)),''));
                    end
                case 2
            end
            NameStr = strjoin(string(str_name),'');
        end
    end
    
    % Static methods
    methods(Static)
        function saveProperties(block,savedir)
            if exist(savedir,'dir')==0
                mkdir(savedir);
            end
            savedir = strcat(savedir,block.MakeName(),'.mat');
            if exist(savedir,'file')==0
                save(savedir,'block');
            else
                fprintf(strcat('File already exists:\n ',savedir));
            end
        end
        
        function self = loadProperties(savedir)
            data = load(savedir);
            self = data.block;
        end
    end
end