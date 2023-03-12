% concat movies
%%
clear
dpath = 'K:\MJH\SortMS\M342\Diestrus\Sess1\behcam\';
SavePath = [dpath,'ConcatedAvi.avi'];
delete(SavePath)



allf = dir([dpath,'*.avi']);
behloc = zeros(size(allf));
for i =1:length(allf)
    if contains(allf(i).name,'behav')
        behloc(i) = 1;
    end
end
lenbehloc = length(find(behloc==1));
if lenbehloc < 3
    behloc = zeros(size(allf));
    for i = 1:length(allf)
        if contains(allf(i).name,'.avi')
            behloc(i) = 1;
        end
    end
end
lenbehloc = length(find(behloc==1));
disp([num2str(lenbehloc),' Behavior Record is Found'])
%%
avi = VideoWriter(SavePath,'Motion JPEG AVI');
open(avi)
disp('Start Writing AVI Movie to the Same Direction')
for idx = 1:lenbehloc
    disp(['Now Writing ',num2str(idx),'th Video'])
    try
        vidPath = [dpath,'behavCam',num2str(idx),'.avi'];
        mov = VideoReader(vidPath);
    catch
        vidPath = [dpath,num2str(idx-1),'.avi'];
        mov = VideoReader(vidPath);
    end
    while hasFrame(mov)
        frame = readFrame(mov);
        writeVideo(avi,frame)
    end
end
close(avi)