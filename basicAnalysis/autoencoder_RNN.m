% This script uses autoencoder to reduce the dimension and use Bi-LSTM to
% classify the 6-dimension hidden time series reponse;
% Train-Valid-Test
clear
close all
all_com = {}; shuffle_lab = {}; 
all_lab = {}; labs = [];
home = 'K:\MJH\MSRes\M';
tsne = 0;
for statex = 1:3
    if statex == 1
        animal = {'893' '352' '342' '882' '886' '955' '959' };
        state = {'Diestrus' };
        sess = { {'1' '4'} {'1'} {'1'} {'3'} {'1', '2'} {'2' '3'} {'1'}} ; %882-4
        cstring = 'b';
        lab = 1;
    elseif statex == 2
        animal = {'900' '910' '918' '970' '65' '82' '90' '91' '92'};
        state = {'Male' };
        sess = { {'5','9', '10'} {'2','5','6'} {'1'} {'1' '2' '3'} {'1' '2'} {'1'} {'1'}  {'1'} {'1'} } ;
        cstring = 'k';
        lab = 2;
    else
        animal = {'342' '352' '451' '882' '883' '954' '955' '959' '983' };
        state = {'Estrus' };
        sess = { {'1','2'} {'1', '2'} {'1'} {'1','4'} {'1'} {'1'} {'1'} {'1' '2'} {'1'}} ;%882-3
        cstring = 'r';
        lab = 3;
    end
    preonset = 0;
    afteronset = 180;

    for i = 1:length(animal)
        for j = 1:length(state)
            session = sess{i};
            for k = 1:length(session)
                dpath = [home,animal{i},'\',state{j},'\Sess',session{k}];

                if exist([dpath,'Res'],'dir')==0
                    continue
                else
                    load([dpath,'Res\PETH.mat'])
                    elab = PETH.elab;
                    NeuTraceMat = PETH.NeuTraceMat;
                    disp([animal{i},' session ',session{k} ' has ',num2str(size(NeuTraceMat,1)), ' neurons'])
                    if size(NeuTraceMat,1) < 15
                        continue
                    end
                    Normalized = func_CalcDeltaf(NeuTraceMat,1,length(NeuTraceMat));
                    NeuTraceMat = matsmooth(Normalized,10);

                    [mtrial,ltrial] = func_getTrialData(NeuTraceMat,elab,60,60);
                    if length(mtrial)<5
                        continue
                    end

                    combinedmat = [];
                    rid = randsample(length(mtrial),length(mtrial));
                    for ix = 1:length(rid)
                        autoenc = trainAutoencoder(mtrial{ix},6,'MaxEpochs',3500);
                        if lab==3
                            all_com{end+1} = 1.25*autoenc.encode(mtrial{ix});
                        else
                            all_com{end+1} = autoenc.encode(mtrial{ix});
                        end
                        all_lab{end+1} = lab;
                        shuffle_lab{end+1} = randi(3,1,1);
                        labs = [labs lab];
                    end
                end
            end
        end
    end
end
%% Train RNN (Bi-LSTM)
tF1 = [];
tAccu = [];
control = 1;
val_num = 24;
for iter = 1:50
    disp(iter)
    val_id = [randsample(1:max(find(labs==1)),val_num),randsample(max(find(labs==1))+1:max(find(labs==2)),val_num),randsample(max(find(labs==2))+1:max(find(labs==3)),val_num)];
    testid = [randsample(val_id(1:val_num),val_num/2),randsample(val_id(val_num+1:2*val_num),val_num/2),randsample(val_id(val_num*2+1:val_num*3),val_num/2)];
    vid = setdiff(val_id , testid);
    tid = setdiff(1:length(all_com),vid);
    XTrain = all_com(tid);
    YTrain = [];
    for ixx = 1:length(tid)
        labtid = all_lab(tid);
        YTrain = [YTrain;categorical(labtid{ixx})];
    end
    XVal = all_com(vid);
    YVal = [];
    for ixx = 1:length(vid)
        labvid = all_lab(vid);
        YVal = [YVal;categorical(labvid{ixx})];
    end
    if control
        shuffleid = randperm(length(YTrain));
        YTrain = YTrain(shuffleid);
    end
    inputSize = 6;
    numHiddenUnits = 256;
    numClasses = 3;

%     layers = [ ...
%         sequenceInputLayer(inputSize)
%         bilstmLayer(numHiddenUnits,'OutputMode','last')
%         fullyConnectedLayer(numClasses)
%         softmaxLayer
%         classificationLayer];
    layers = [
        sequenceInputLayer(inputSize,"Name","SeqIn")
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(0.1,"Name","dropout")
        reluLayer("Name","relu")        
        fullyConnectedLayer(3,"Name","fc")
        softmaxLayer("Name","softmax")
        classificationLayer];
    maxEpochs = 200;
    miniBatchSize = 10;

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'ValidationData',{XVal,YVal}, ...
        'ValidationFrequency',20, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'GradientThreshold',1, ...
        'Verbose',0, ...
        'Plots','none', ...
        'OutputNetwork','best-validation-loss');
    [net,info] = trainNetwork(XTrain,YTrain,layers,options);

    if max(info.ValidationAccuracy)>90
        save('.\bestmodel.mat','net')
    end
    netpred = predict(net,all_com(testid));
    [~,netpredlab] = max(netpred,[],2);
    Yt = [];
    for ixx = 1:length(testid)
        labvid = all_lab(testid);
        Yt = [Yt;labvid{ixx}];
    end
    testaccu = mean(Yt == netpredlab);
    [f1,f1_es] = f1_score(Yt,netpredlab,3);
    tF1 = [tF1 f1_es];
    tAccu = [tAccu testaccu]
end
nanmean(tAccu)
%send_email(['Your Accuracy is ',num2str(mean(tAccu)),', Your F1 is ',num2str(mean(tF1))])