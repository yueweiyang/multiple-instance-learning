clear all;
close all;

plotdb = 0;
nDataSet = 10;
Percent = .5;
PosPortion = [1,5:5:50];
Sigma = [0.01:0.01:0.2]%[.01:.04:.1,.1:.2:0.9,1,1.1:.2:2,2:.5:4];
MdlDir = '/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/mdl/noise0.30/';

AUC_miSVM_t = 0;
AUC_SVM_t = 0;

for ndataset = 1:nDataSet
    AUC_SVM = zeros(length(Sigma),length(PosPortion));
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
            AUC_SVM(nsigma,nposportion) = AUC(1);
            AUC_miSVM(nsigma,nposportion) = AUC(end);
        end
    end
    AUC_miSVM_t = AUC_miSVM_t+AUC_miSVM;
    AUC_SVM_t = AUC_SVM_t+AUC_SVM;
end

AUC_miSVM = AUC_miSVM_t./nDataSet;
AUC_SVM = AUC_SVM_t./nDataSet;

figure;
subplot(1,2,1);
imagesc(PosPortion/50,Sigma,AUC_SVM);
colorbar;
caxis([0.5,1]);
xlim([min(PosPortion)/50,max(PosPortion)/50]);
ylim([min(Sigma),max(Sigma)]);
yticks([1:2:length(Sigma)]/(length(Sigma)/max(Sigma)));
yticklabels(string(Sigma(1:2:length(Sigma))));
xlabel('\alpha');
ylabel('\sigma');
title('SVM');
set(gca,'fontsize',20);
subplot(1,2,2);
imagesc(PosPortion/50,Sigma,AUC_miSVM);
colorbar;
caxis([0.5,1]);
xlim([min(PosPortion)/50,max(PosPortion)/50]);
ylim([min(Sigma),max(Sigma)]);
yticks([1:2:length(Sigma)]/(length(Sigma)/max(Sigma)));
yticklabels(string(Sigma(1:2:length(Sigma))));
xlabel('\alpha');
ylabel('\sigma');
title('miSVM');
set(gca,'fontsize',20);

figure;
subplot(1,2,1);
imagesc(PosPortion/50,Sigma,(AUC_miSVM-AUC_SVM));
colorbar;
xlim([min(PosPortion)/50,max(PosPortion)/50]);
ylim([min(Sigma),max(Sigma)]);
yticks([1:2:length(Sigma)]/(length(Sigma)/max(Sigma)));
yticklabels(string(Sigma(1:2:length(Sigma))));
xlabel('\alpha');
ylabel('\sigma');
set(gca,'fontsize',20);
subplot(1,2,2);
imagesc(PosPortion/50,Sigma,sign(AUC_miSVM-AUC_SVM));
c = colorbar;
caxis([-1,1]);
c.Ticks = [-1,1];
c.TickLabels = {'miSVM worse','miSVM better'};
xlim([min(PosPortion)/50,max(PosPortion)/50]);
ylim([min(Sigma),max(Sigma)]);
yticks([1:2:length(Sigma)]/(length(Sigma)/max(Sigma)));
yticklabels(string(Sigma(1:2:length(Sigma))));
xlabel('\alpha');
ylabel('\sigma');
set(gca,'fontsize',20);


MaxInd_miSVM = AUC_miSVM==(max(AUC_miSVM));
OptSigma_miSVM = Sigma*MaxInd_miSVM./sum(MaxInd_miSVM);
MaxInd_SVM = AUC_SVM==(max(AUC_SVM));
OptSigma_SVM = Sigma*MaxInd_SVM./sum(MaxInd_SVM);

figure;
plot(PosPortion/50,OptSigma_miSVM,'-*',PosPortion/50,OptSigma_SVM,...
    '-*','LineWidth',4,'MarkerSize',10);
legend('miSVM','SVM');
xlabel('\alpha');
ylabel('\sigma');
set(gca,'fontsize',20);

figure;
plot(PosPortion/50,(max(AUC_miSVM)-max(AUC_SVM))./max(AUC_SVM),...
    '-*','LineWidth',4,'MarkerSize',10);
xlabel('\alpha');
ylabel('\sigma');
set(gca,'fontsize',20);

