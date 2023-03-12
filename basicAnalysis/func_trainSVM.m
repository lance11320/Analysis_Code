function [tw_accu,tw_con_accu,model] = func_trainSVM(NeuTraceMat,elab,params)
[SniMStart,SniMEnd,SniLStart,SniLEnd] = getSniffSE(elab);
%% split the data set for normal model training
Mlen = length(SniMStart); Llen = length(SniLStart);
% different time epoch training set
preonset = params.preonset;
afteronset = params.afteronset;
NeuronNum = params.neunum;
binsize = params.binsize;
stepsize = params.stepsize;
tw_accu = []; tw_con_accu = [];
for pre_iter = 1:100
    randid = randsample(size(NeuTraceMat,1),NeuronNum);
    RandTraceMat = NeuTraceMat(randid,:);
    OnsetTraceA = {};
    tnum = 0;
    for tid = 1:Mlen
        if tid == 1 || SniMStart(tid)>SniMEnd(tid-1)+120
            try
                OnsetAct = RandTraceMat(:,SniMStart(tid)-preonset:SniMStart(tid)+afteronset);
                OnsetTraceA{end+1} = OnsetAct;
                tnum = tnum + 1;
            catch
                continue
            end
        end
    end
    time_xtrainM = {};
    for idx = 1:stepsize:preonset+afteronset-binsize
        xtrain = [];
        for ii = randsample(1:length(OnsetTraceA),5)
            xii = OnsetTraceA{ii}(:,idx:idx+binsize-1);
            xtrain = [xtrain xii];
        end
        time_xtrainM{end+1} = xtrain;
    end

    OnsetTraceA = {};
    for tid = 1:Llen
        if tid == 1 || SniLStart(tid)>SniLEnd(tid-1)
            try
                OnsetAct = RandTraceMat(:,SniLStart(tid)-preonset:SniLStart(tid)+afteronset);
                OnsetTraceA{end+1} = OnsetAct;
            catch
                continue
            end
        end
    end
    time_xtrainL = {};
    for idx = 1:stepsize:preonset+afteronset-binsize
        xtrain = [];
        for ii = randsample(1:length(OnsetTraceA),5)
            xii = OnsetTraceA{ii}(:,idx:idx+binsize-1);
            xtrain = [xtrain xii];
        end
        time_xtrainL{end+1} = xtrain;
    end
    accu_time = []; shu_accu_time = [];

    for time_i = 1:length(time_xtrainM)
        
        mtrain = time_xtrainM{time_i};
        ltrain = time_xtrainL{time_i};
        stid = 1:binsize:5*binsize; edid = binsize:binsize:5*binsize;
        ac_time = []; shu_ac_time = [];
        for cviter = 1:5
            testid = stid(cviter):edid(cviter);
            trainid = setdiff(1:5*binsize,testid);
            x_train = [ltrain(:,trainid) mtrain(:,trainid)];
            y_train = [zeros(1,length(trainid)),ones(1,length(trainid))];
            y_shu_id = randperm(length(y_train)); y_rd = y_train(y_shu_id);
            x_test = [ltrain(:,testid) mtrain(:,testid)];
            y_test = [zeros(1,length(testid)),ones(1,length(testid))];
            model = svmtrain(y_train',x_train','-t 0 -c 2 -q');
            model_shu = svmtrain(y_rd',x_train','-t 0 -c 2 -q');

            pred = svmpredict(y_test',x_test',model);
            accu = mean(y_test' == pred);

            ac_time = [ac_time accu];

            pred_shu = svmpredict(y_test',x_test',model_shu);
            con_accu = mean(y_test' == pred_shu);
            shu_ac_time = [shu_ac_time con_accu];
        end
        accu_time = [accu_time mean(ac_time)];
        shu_accu_time = [shu_accu_time mean(shu_ac_time)];

    end

    tw_accu = [tw_accu ;accu_time];
    tw_con_accu = [tw_con_accu ; shu_accu_time];
end
tw_accu = mean(tw_accu);
tw_con_accu = mean(tw_con_accu);

end