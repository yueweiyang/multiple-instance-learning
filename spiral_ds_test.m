ds = prtDataGenSpiral(1000/2);
angle = 30;
ds.data(ds.targets==1,:) = ds.data(ds.targets==1,:)*[cos(angle),-sin(angle);sin(angle),cos(angle)]+[0,0.5];
ds.targets;
plot(ds);