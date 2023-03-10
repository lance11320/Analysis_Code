% Modified code for ReadAviAndndRegister
% Only suit for working station
function [NeuTraceMat] = func_GetTraceFast(LocMat,filename)

%% Load
UnitLen = size(LocMat,1);
mov = VideoReader(filename);
FrameNum = mov.Duration*mov.FrameRate;
FrHeight = mov.Height;
FrWidth = mov.Width;
frame = read(mov);
Vdata = squeeze(frame(:,:,1,:));% squeeze to Height.Width.Frame
allV = Vdata;%(:,753:1504,:);
%NeuTraceMat = zeros(UnitLen,FrameNum);
%% Calc
tic

%LocMat(LocMat~=0)=1;
%LocMat(LocMat~=1)=0;
[k1,k2,k3] = size(allV);
[t1,t2,t3] = size(LocMat);
allvv = reshape(allV,k1*k2,k3);%turn H.W matrix to vector
lm = reshape(LocMat,t1,t2*t3);%turn H.W matrix to vector
res = lm*im2double(allvv);%0-1 matrix multiply
var = sum(lm,2);
var = repmat(var,1,k3);
NeuTraceMat = res./var;
%NeuTraceMat = im2uint8(NeuTraceMat);
NeuTraceMat = 255*im2double(NeuTraceMat);
toc
return