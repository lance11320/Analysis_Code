% Using engineered features to classify
clear
close all
all_com = {};
all_lab = {};
ani_di = []; ani_es = []; ani_m = [];
home = 'K:\MJH\MSRes\M';
warning off
all_flag = 1;
ramp_flag = 0;
frac_flag = 0;
resp_flag = 0;
d_flag = 0;
md_flag = 0;

if all_flag == 1
    ramp_flag = 0; frac_flag = 1; resp_flag = 1; d_flag = 1; md_flag = 0;
end

for statex = 1:3
    if statex == 1
        animal = {'893' '882' '886' '955' '959'};
        state = {'Diestrus' };
        sess = { {'1' '4'} {'3','4'} {'1', '2'} {'2' '3'} {'1'} } ;
        train_di = {};
        cstring = 'b';
        lab = 1;
    elseif statex == 2
        animal = {'900' '910' '918' '970' '65' '82' '90' '91' '92'};
        state = {'Male' };
        sess = { {'5','9'} {'2','5'} {'1'} {'1' '2' '3'} {'1' '2'} {'1'} {'1'} {'1'} {'1'}} ;
        train_m = {};
        cstring = 'k';
        lab = 2;
    else
        animal = {'342' '352' '451' '882' '954' '955' '959' '983' };% '342' '352' {'1','2'} {'1', '2'}
        state = {'Estrus' };
        sess = { {'1','2'} {'1', '2'} {'1'} {'1' '4'} {'1'} {'1'} {'1' '2'} {'1'} };%882-3
        train_es = {};
        cstring = 'r';
        lab = 3;
    end

    for i = 1:length(animal)
        for j = 1:length(state)
            session = sess{i};
            for k = 1:length(session)
                dpath = [home,animal{i},'\',state{j},'\Sess',session{k},'Res\'];
               
                if exist(dpath,'dir')==0
                    continue
                else
                    load([dpath,'PETH.mat'])
                    elab = PETH.elab;
                    NeuTraceMat = PETH.NeuTraceMat;
                    disp([animal{i},' session ',session{k} ' has ',num2str(size(NeuTraceMat,1)), ' neurons'])

                    if size(NeuTraceMat,1) < 15
                        disp('Less Than 15 Neurons, Pass')
                        continue
                    end

                    Normalized = func_CalcDeltaf(NeuTraceMat,1,length(NeuTraceMat));
                    NeuTraceMat = matsmooth(Normalized,10);

                    preonset = 120; afteronset = 120;
                    [mtrial,ltrial] = func_getTrialData(NeuTraceMat,elab,preonset,afteronset);
                    if length(mtrial)<5
                        disp('Less Than 5 Trials, Pass')
                        continue    
                    end
                    if statex == 1
                        for trialsnum = 1:length(mtrial)
                            train_di{end+1} = mtrial{trialsnum};
                            ani_di(end+1) = i;
                        end
                    elseif statex == 2
                        for trialsnum = 1:length(mtrial)
                            train_m{end+1} = mtrial{trialsnum};
                            ani_m(end+1) = i;
                        end
                    else
                        for trialsnum = 1:length(mtrial)
                            train_es{end+1} = mtrial{trialsnum};
                            ani_es(end+1) = i;
                        end
                    end

                end

            end
        end
    end
end
%% extract features

%% Distribution of response time (min time) (contribute1%)
start_di = []; start_es = []; start_m = [];
low_di = []; low_es = []; low_m = [];
if resp_flag == 1

for ix = 1:length(train_di)
    wave_s = [];
    for id  = 1:size(train_di{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_di{ix}(id,:),1);
        if auc>0.7
            [pks_min,loc_max] = findpeaks(10-train_di{ix}(id,1:240));
            minloc = 1+loc_max(pks_min==max(pks_min));
        end
        wave_s = [wave_s minloc];
    end
    low_di = [low_di mean(wave_s)];
end
for ix = 1:length(train_es)
    wave_s = [];
    for id  = 1:size(train_es{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_es{ix}(id,:),1);
        if auc>0.7
            [pks_min,loc_max] = findpeaks(10-train_es{ix}(id,1:240));
            minloc = 1+loc_max(pks_min==max(pks_min));
        end
        wave_s = [wave_s minloc];
    end
    low_es = [low_es mean(wave_s)];
end
for ix = 1:length(train_m)
    wave_s = [];
    for id  = 1:size(train_m{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_m{ix}(id,:),1);
        if auc>0.7
            [pks_min,loc_max] = findpeaks(10-train_m{ix}(id,1:240));
            minloc = 1+loc_max(pks_min==max(pks_min));
        end
        wave_s = [wave_s minloc];
    end
    low_m = [low_m mean(wave_s)];
end
%% Distribution of response time (max time) (contribute1%)

for ix = 1:length(train_di)
    wave_s = [];
    for id  = 1:size(train_di{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_di{ix}(id,:),1);
        if auc>0.7
            [pks_max,loc_max] = findpeaks(train_di{ix}(id,80:180));
            maxloc = loc_max(pks_max==max(pks_max));
            [pks,locs] = findpeaks(10-train_di{ix}(id,80:80+maxloc)); %local minimum
            minloc = locs(pks == max(pks));
            start = minloc+80; peakloc = maxloc+80;
        end
        wave_s = [wave_s peakloc];
    end
    start_di = [start_di mean(wave_s)];
end

for ix = 1:length(train_es)
    wave_s = [];
    for id  = 1:size(train_es{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_es{ix}(id,:),1);
        if auc>0.7
            [pks_max,loc_max] = findpeaks(train_es{ix}(id,80:180));
            maxloc = loc_max(pks_max==max(pks_max));
            [pks,locs] = findpeaks(10-train_es{ix}(id,80:80+maxloc)); %local minimum
            minloc = locs(pks == max(pks));
            start = minloc+80; peakloc = maxloc+80;
        end
        wave_s = [wave_s peakloc];
    end
    start_es = [start_es mean(wave_s)];
end
for ix = 1:length(train_m)
    wave_s = [];
    for id  = 1:size(train_m{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_m{ix}(id,:),1);
        if auc>0.7
            [pks_max,loc_max] = findpeaks(train_m{ix}(id,80:180));
            maxloc = loc_max(pks_max==max(pks_max));
            [pks,locs] = findpeaks(10-train_m{ix}(id,80:80+maxloc)); %local minimum
            minloc = locs(pks == max(pks));
            start = minloc+80; peakloc = maxloc+80;
        end
        wave_s = [wave_s peakloc];
    end
    start_m = [start_m mean(wave_s)];
end

end

%% Diff of activated neuron (contribute3%)
dif_di = []; dif_m = []; dif_es = [];
if d_flag == 1
for ix = 1:length(train_di)
    Dif = [];
    for id  = 1:size(train_di{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_di{ix}(id,:),1);
        if auc>0.7
            diff = mean(train_di{ix}(id,120:180)) - mean(train_di{ix}(id,60:120));
        end
        Dif = [Dif diff];
    end
    dif_di = [dif_di mean(Dif)];
end
for ix = 1:length(train_es)
    Dif = [];
    for id  = 1:size(train_es{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_es{ix}(id,:),1);
        if auc>0.7
            diff = mean(train_es{ix}(id,120:180)) - mean(train_es{ix}(id,60:120));
        end
        Dif = [Dif diff];
    end
    dif_es = [dif_es mean(Dif)];
end
for ix = 1:length(train_m)
    Dif = [];
    for id  = 1:size(train_m{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_m{ix}(id,:),1);
        if auc>0.7
            diff = mean(train_m{ix}(id,120:180)) - mean(train_m{ix}(id,60:120));
        end
        Dif = [Dif diff];
    end
    dif_m = [dif_m mean(Dif)];
end

end

%% AUC 
exc_di = [];
inh_di = [];
exc_m = [];
inh_m = [];
exc_es = [];
inh_es = [];
if frac_flag == 1
for ix = 1:length(train_di)
    exc = 0; inh = 0;
    for id = 1:size(train_di{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_di{ix}(id,:),1);
        if auc>0.7
            exc = exc+1;
        elseif auc<0.3
            inh = inh+1;
        end
    end
    exc_f = exc/size(train_di{ix},1);
    inh_f = inh/size(train_di{ix},1);
    exc_di = [exc_di exc_f];
    inh_di = [inh_di inh_f];
end

for ix = 1:length(train_m)
    exc = 0; inh = 0;
    for id = 1:size(train_m{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_m{ix}(id,:),1);
        if auc>0.7
            exc = exc+1;
        elseif auc<0.3
            inh = inh+1;
        end
    end
    exc_f = exc/size(train_m{ix},1);
    inh_f = inh/size(train_m{ix},1);
    exc_m = [exc_m exc_f];
    inh_m = [inh_m inh_f];
end


for ix = 1:length(train_es)
    exc = 0; inh = 0;
    for id = 1:size(train_es{ix},1)
        [X,Y,T,auc] = perfcurve([zeros(1,120),ones(1,121)],train_es{ix}(id,:),1);
        if auc>0.7
            exc = exc+1;
        elseif auc<0.3
            inh = inh+1;
        end
    end
    exc_f = exc/size(train_es{ix},1);
    inh_f = inh/size(train_es{ix},1);
    exc_es = [exc_es exc_f];
    inh_es = [inh_es inh_f];
end

end

%% SVM
Accu = [];
F1 = [];
randx = 60;randy = 60;randz = 60;
whole = randx + randy + randz;
twoclass = 0;
if twoclass == 1
    labind = 1;
else
    labind = 2;
end
for iter = 1:500
x_es = [corr_es;exc_es;inh_es;dif_es;start_es;low_es;ramp_es;mdif_es]; rid1 = randsample(length(x_es),randx);
x_di = [corr_di;exc_di;inh_di;dif_di;start_di;low_di;ramp_di;mdif_di]; rid2 = randsample(length(x_di),randy);
x_m = [corr_m;exc_m;inh_m;dif_m;start_m;low_m;ramp_m;mdif_m]; rid3 = randsample(length(x_m),randz);
x = mapminmax([x_es(:,rid1) x_di(:,rid2) x_m(:,rid3)],0,1);
y = [zeros(randx,1);ones(randy,1);labind*ones(randz,1)];
accu = svmtrain(y,x','-t 0 -c 2 -v 5 -q');
Accu = [Accu accu];

for round = 1:5
[trainid, testid] = crossvalind('LeaveMOut',whole,3*whole/5);
x = mapminmax([x_es(:,rid1) x_di(:,rid2) x_m(:,rid3)],0,1);
y = [zeros(randx,1);ones(randy,1);labind*ones(randz,1)];
xtrain = x(:,trainid); ytrain = y(trainid);
xtest = x(:,testid); ytest = y(testid);
model = svmtrain(ytrain,xtrain','-t 0 -c 5 -q');
pred = svmpredict(ytest,xtest',model);
[f1,f1_es] = f1_score(ytest,pred);
F1 = [F1 f1_es];
end
end
disp(['Mean CV Accuracy is ',num2str(mean(Accu))])
disp(['Mean F1 is ',num2str(nanmean(F1))])
%roc_curve(pred',ytest')

%% shuffle control;
shu_accu = []; Shu = []; F1_s = []; F1_shu = [];
if twoclass == 1
    labind = 1;
end
for iter = 1:500
x_es = [corr_es;exc_es;inh_es;dif_es;start_es;low_es;ramp_es;mdif_es]; rid1 = randsample(length(x_es),randx);
x_di = [corr_di;exc_di;inh_di;dif_di;start_di;low_di;ramp_di;mdif_di]; rid2 = randsample(length(x_di),randx);
x_m = [corr_m;exc_m;inh_m;dif_m;start_m;low_m;ramp_m;mdif_m]; rid3 = randsample(length(x_m),randx);
for round = 1:5
[trainid, testid] = crossvalind('LeaveMOut',3*randx,3/5*randx);
x = mapminmax([x_es(:,rid1) x_di(:,rid2) x_m(:,rid3)],0,1);
y = [zeros(randx,1);ones(randx,1);labind*ones(randx,1)];
xtrain = x(:,trainid); ytrain = y(trainid);
xtest = x(:,testid); ytest = y(testid);

y_shu = randi([0 labind],size(ytrain));
model_shu = svmtrain(y_shu,xtrain','-t 0 -c 2 -q');
pred_shu = svmpredict(ytest,xtest',model_shu);
shu_accu = [shu_accu mean(pred_shu==ytest)];
f1_ = f1_score(ytest,pred_shu);
F1_s = [F1_s f1_];
end
Shu = [Shu mean(shu_accu)];
F1_shu = [F1_shu mean(F1_s)];
end


%% SVM 2-class
%es_di
es_di_accu = [];
for iter = 1:500
x_es = [corr_es;exc_es;inh_es;dif_es;start_es;low_es;ramp_es;mdif_es]; rid1 = randsample(length(x_es),randx);
x_di = [corr_di;exc_di;inh_di;dif_di;start_di;low_di;ramp_di;mdif_di]; rid2 = randsample(length(x_di),randx);
x_m = [corr_m;exc_m;inh_m;dif_m;start_m;low_m;ramp_m;mdif_m]; rid3 = randsample(length(x_m),randx);
x = mapminmax([x_es(:,rid1) x_di(:,rid2)],0,1);
y = [zeros(randx,1);ones(randy,1)];
accu = svmtrain(y,x','-t 0 -c 2 -v 5 -q');
es_di_accu = [es_di_accu accu];
end

%es_m
es_m_accu = [];
for iter = 1:500
x_es = [corr_es;exc_es;inh_es;dif_es;start_es;low_es;ramp_es;mdif_es]; rid1 = randsample(length(x_es),randx);
x_di = [corr_di;exc_di;inh_di;dif_di;start_di;low_di;ramp_di;mdif_di]; rid2 = randsample(length(x_di),randx);
x_m = [corr_m;exc_m;inh_m;dif_m;start_m;low_m;ramp_m;mdif_m]; rid3 = randsample(length(x_m),randx);
x = mapminmax([x_es(:,rid1) x_m(:,rid3)],0,1);
y = [zeros(randx,1);ones(randy,1)];
accu = svmtrain(y,x','-t 0 -c 2 -v 5 -q');
es_m_accu = [es_m_accu accu];
end

%m_di
di_m_accu = [];
for iter = 1:500
x_es = [corr_es;exc_es;inh_es;dif_es;start_es;low_es;ramp_es;mdif_es]; rid1 = randsample(length(x_es),randx);
x_di = [corr_di;exc_di;inh_di;dif_di;start_di;low_di;ramp_di;mdif_di]; rid2 = randsample(length(x_di),randx);
x_m = [corr_m;exc_m;inh_m;dif_m;start_m;low_m;ramp_m;mdif_m]; rid3 = randsample(length(x_m),randx);
x = mapminmax([x_di(:,rid1) x_m(:,rid3)],0,1);
y = [zeros(randx,1);ones(randy,1)];
accu = svmtrain(y,x','-t 0 -c 2 -v 5 -q');
di_m_accu = [di_m_accu accu];
end

disp(['The CV Accuracy of ES-Di is ',num2str(mean(es_di_accu))])
disp(['The CV Accuracy of ES-M is ',num2str(mean(es_m_accu))])
disp(['The CV Accuracy of Di-M is ',num2str(mean(di_m_accu))])

disp(['Mean CV Accuracy is ',num2str(mean(Accu))])
disp(['Mean F1 is ',num2str(nanmean(F1))])
disp(['Mean CV Accuracy with Shuffle is ',num2str(mean(Shu))])
disp(['Mean F1 with Shuffle is ',num2str(mean(F1_shu))])


%%
diffea = mapminmax([dif_es,dif_di,dif_m],0,1);
startfea = mapminmax([start_es,start_di,start_m],0,1);
lowfea = mapminmax([low_es,low_di,low_m],0,1);
excfea = mapminmax([exc_es,exc_di,exc_m],0,1);
inhfea = mapminmax([inh_es,inh_di,inh_m],0,1);


function [score,score1] = f1_score(label, predict,varargin)
if isempty(varargin)
    ind = 1;
else
    ind = varargin{1};
end

   M = confusionmat(label, predict);
   M = M';
   precision = diag(M)./(sum(M,2) + 0.0001);  
   recall = diag(M)./(sum(M,1)+0.0001)';
   score1 = 2*precision(ind)*recall(ind)/(precision(ind) + recall(ind));
   precision = mean(precision);
   recall = mean(recall);
   score = 2*precision*recall/(precision + recall);
end

