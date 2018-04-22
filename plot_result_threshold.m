clear all;
close all;

plotdb = 0;
nDataSet = 30;
Percent = .5;
PosPortion = [1,5:5:50];
Sigma = [0.09,1,2,3];
Threshold = [0.05,0.1,0.5,1,1.5,2];
Step = [0.2,0.5,1];
MdlDir = '/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/mdl/';

AUC_miSVM_t = 0;

for ndataset = 1:nDataSet
    AUC_miSVM = zeros(length(Sigma),length(PosPortion));
    for nposportion = 1:length(PosPortion)
        posportion = PosPortion(nposportion);
        for nsigma = 1:length(Sigma)
            sigma = Sigma(nsigma);
            str_filename = strcat(MdlDir,...
                sprintf('DataSet%d/Percent%d/Sigma%.2f/PosPortion%d.mat',...
                ndataset,Percent*100,sigma,posportion));
            mdl = block_mdl.loadProperties(str_filename);
            if ndataset == 1 && plotdb
                mdl.PlotDecBound();
            end
            AUC = nonzeros(mdl.Result.AUC);
            AUC_miSVM(nsigma,nposportion) = AUC(end);
        end
    end
    AUC_miSVM_t = AUC_miSVM_t+AUC_miSVM;
end

AUC_miSVM = AUC_miSVM_t./nDataSet;

pic_row = length(Threshold);
pic_col = length(Sigma);
for nsigma = 1:length(Sigma)
    sigma = Sigma(nsigma);
    for nthreshold = 1:length(Threshold)
        threshold = Threshold(nthreshold);
        AUC_th_miSVM_t = 0;
        for ndataset = 1:nDataSet
            AUC_th_miSVM = zeros(length(Step),length(PosPortion));
            for nstep = 1:length(Step)
                step = Step(nstep);
                for nposportion = 1:length(PosPortion)
                    posportion = PosPortion(nposportion);
                    str_filename = strcat(MdlDir,...
                        sprintf('DataSet%d/Percent%d/Threshold%.2f/Step%.2f/Sigma%.2f/PosPortion%d.mat',...
                        ndataset,Percent*100,threshold,step,sigma,posportion));
                    mdl = block_mdl.loadProperties(str_filename);
                    AUC = nonzeros(mdl.Result.AUC);
                    AUC_th_miSVM(nstep,nposportion) = AUC(end);
                end
            end
            AUC_th_miSVM_t = AUC_th_miSVM_t+AUC_th_miSVM;
        end
        figure(5);
        subplot(pic_row,pic_col,(nthreshold-1)*length(Sigma)+nsigma);
        plot(transpose([AUC_th_miSVM_t./nDataSet;AUC_miSVM(nsigma,:)]),...
            'LineWidth',4);
        xlim([1,length(PosPortion)]);
        legend('step .2','step .5','step 1',sprintf('sigma%.2f',sigma));
        if nthreshold == 1
            title(sprintf('Sigma%.2f',sigma));
        end
        if nsigma == 1
            ylabel(sprintf('Threshold%.2f \nAUC',threshold));
        else
            ylabel('AUC');
        end
        xlabel('\alpha');
        set(gca,'fontsize',12);
        figure(6);
        subplot(pic_row,pic_col,(nthreshold-1)*length(Sigma)+nsigma);
        plot(transpose([AUC_th_miSVM_t./nDataSet-...
            repmat(AUC_miSVM(nsigma,:),length(Step),1)]./...
            repmat(AUC_miSVM(nsigma,:),length(Step),1)),'LineWidth',4);
        xlim([1,length(PosPortion)]);
        legend('step .2','step .5','step 1');
        if nthreshold == 1
            title(sprintf('Sigma%.2f',sigma));
        end
        if nsigma == 1
            ylabel(sprintf('Threshold%.2f \nAUC',threshold));
        else
            ylabel('AUC');
        end
        xlabel('\alpha');
        set(gca,'fontsize',12);
    end
end