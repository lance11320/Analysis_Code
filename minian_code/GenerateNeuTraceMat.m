%%
clear
animal = {'65'};%
sess = { {'1'} };
state = {'Male'};
% animal = {'910','900','970','38','65','983','882','886','955'};
% state = {'Male','Estrus','Diestrus'};
% sess = {{'6'},{'10'},{'2','3'},{'1'},{'1'},{'1'},{'4'},{'2'},{'3'}} ;
% animal = {'893','959','910','900','970','38','65','983','882','886','955'};
% state = {'Male','Estrus','Diestrus'};
% sess = {{'4'}, {'2'},{'6'},{'10'},{'2','3'},{'1'},{'1'},{'1'},{'4'},{'2'},{'3'}} ;

home = 'J:\MJH\SortMS_EsDi\';
for ii = 1:length(animal)
    for j = 1:length(state)
        session = sess{ii};
        for k = 1:length(session)
            dpath = [home,'M',animal{ii},'\',state{j},'\Sess',session{k},'Res'];
            if exist(dpath,'dir')
                disp(['Now Processing ',dpath])
                % write new video (motion corrected) and use the video to
                % extract
                SaveVidPath = [dpath,'\minian_all.avi'];
                file1 = load([dpath,'\varr1']);
                f1len = size(file1.array,1);
                vid = VideoWriter(SaveVidPath,'Grayscale AVI');
                open(vid)
                for i = 1:f1len
                    writeVideo(vid,squeeze(file1.array(i,:,:)))
                end
                disp('file1 finished')
                file2 = load([dpath,'\varr2']);
                f2len = size(file2.array,1);
                for i = 1:f2len
                    writeVideo(vid,squeeze(file2.array(i,:,:)))
                end
                disp('file2 finished')
                for idx = 3:10
                    
                    if exist([dpath,'\varr',num2str(idx),'.mat'],'file')==2
                        disp(['file',num2str(idx),'detected'])
                        file3 = load([dpath,'\varr',num2str(idx)]);
                        f3len = size(file3.array,1);
                        for i = 1:f3len
                            writeVideo(vid,squeeze(file3.array(i,:,:)))
                        end
                    end
                end
                close(vid)
                disp('done')
                try
                    Apath = [dpath,'\',animal{ii},'Chose_A.mat'];
                    LocMat = load(Apath); LocMat = LocMat.newarea;
                    disp('Using Manually Selected Footprint to Extract')
                catch
                    Apath = [dpath,'\',animal{ii},'A.mat'];
                    LocMat = load(Apath); LocMat = LocMat.array;
                    disp('Using minian Selected Footprint to Extract')
                end
                
                %extract trace fast; only suit for work station with large
                %memory
                NeuTraceMat = func_GetTraceFast(LocMat,SaveVidPath); 
                
                save([dpath,'\','NeuTrace.mat'],'NeuTraceMat')
                disp(['Saved to ',dpath,'\','NeuTrace.mat'])

            end
        end
    end
end
send_email('Done in converting miniscope data into NeuTraceMat')
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
