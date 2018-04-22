% clear all;
clear all;
close all;
load('/Users/yueweiyang/Documents/study/duke/independent study/fall 2017/matlab/multiple instance learning/result/data/Spiralssize30000nclusters2noise0.00.mat')

% 
% 
% pos = [1:3000];
% neg = [3001:6000];
% 
% pos = pos(randperm(3000));
% neg = neg(randperm(3000));
% 
% train = [pos(1:2500),neg(1:2500)];
% test = [pos(2501:end),neg(2501:end)];
% 
% 
% 
% X =block.Data.X(train,:);
% Y = block.Data.Y(([train]));
% xtest = block.Data.X([test],:);
% ytest = block.Data.Y(test);
% SVMMdl = fitcsvm(X,Y,'KernelFunction','rbf','KernelScale',4);
% 
% [l,s] = predict(SVMMdl,xtest);
% [pd,pf,th,auc] = perfcurve(ytest,s(:,2),1);
% auc
% 
% ds = prtDataSetClass(SVMMdl.X,SVMMdl.Y);
% dstest = prtDataSetClass(xtest,ytest);
% gamma = 1/(SVMMdl.KernelParameters.Scale^2);
% mdl = prtClassLibSvm('kernelType',2,'gamma',gamma);
% mdl_c = mdl.train(ds);
% plot(mdl_c);
% 
% mdl_cc = run(mdl_c,dstest);
% prtScoreAuc(mdl_cc,dstest)



% 
% 
ndata = 30000;

theta = linspace(0,1080,ndata);
rho = 0.005*theta/10;
freq = 5;
angle = 2*pi*freq*(1:ndata);
% angle(end) = [];
am = 5;
rho = rho+am.*sin(angle);
theta_radians = deg2rad(theta);
% polarplot(theta_radians,rho)
[x,y] = pol2cart(theta_radians,rho);

data = block;
data.Data.X = [[x',y'];-[x',y']];
data.PlotData;

posind = 1:30000;
negind = 30001:60000;

posind = posind(randperm(30000));
negind = negind(randperm(30000));

ds = prtDataSetClass([data.Data.X(posind(1:5000),:);data.Data.X(negind(1:5000),:)],[data.Data.Y(posind(1:5000));data.Data.Y(negind(1:5000))]);
dstest = prtDataSetClass([data.Data.X(posind(10000:12000),:);data.Data.X(negind(10000:12000),:)],[data.Data.Y(posind(10000:12000));data.Data.Y(negind(10000:12000))]);
scale = 1;
gamma = 1/(scale^2);
mdl = prtClassLibSvm('kernelType',2,'gamma',gamma);
mdl_trained = mdl.train(ds);
figure();
plot(mdl_trained);

mdl_test = run(mdl_trained,dstest);
prtScoreAuc(mdl_test,dstest)



SVMMdl = fitcsvm(data.Data.X,data.Data.Y,'KernelFunction','rbf','KernelScale',scale);
[l,s] = predict(SVMMdl,dstest.data);
[pd,pf,th,auc] = perfcurve(dstest.targets,s(:,2),1);
