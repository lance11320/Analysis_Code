classdef PETHdata
    properties
        dpath;PETHpath;
        NeuTraceMat; elab;
        cellAUC; cellID;
        trials; timeslides;
        params;
        Intermediate;
        save;
        colorlist;
        beh;
        StartPoint;EndPoint;
        checkpoints;

    end

    methods

        function obj = PETHdata(PETHpath,varargin)
            if ~isempty(varargin)
                home = varargin{1}; root = varargin{2}; animal = varargin{3};
                state = varargin{4}; session = varargin{5};
                obj.dpath = [home,root,animal,'\',state,'\Sess',session];
                obj.save.savepath = [obj.dpath, 'Res\'];
                obj.save.savefig = [obj.dpath, 'Res\SmooUnitTrace\'];
                Arr = load([obj.dpath,'Res\','NeuTrace.mat']);
                obj.NeuTraceMat = Arr.NeuTraceMat;
                [aligned_elab,obj] = func_alignBeh(obj,[home,root],animal,state,session);
                obj.elab = aligned_elab;
                PETH.dpath = obj.dpath;
                PETH.elab = obj.elab(1:length(obj.NeuTraceMat));
                PETH.NeuTraceMat = obj.NeuTraceMat;
                PETHsavepath = [home,'MSRes\M',animal,'\',state,'\Sess',session];
                mkdir(PETHsavepath)
                save([PETHsavepath,'PETH'],'PETH');
            end
            obj.PETHpath = PETHpath;
            load(obj.PETHpath)
            obj.NeuTraceMat = PETH.NeuTraceMat;
            obj.elab = PETH.elab;
        end

        function obj = matsmooth(obj,varargin)
            TraceMat = obj.NeuTraceMat;
            obj.Intermediate.NeuTracemat = obj.NeuTraceMat;
            if isempty(varargin)
                for i = 1:size(TraceMat,1)
                    Trace = smooth(TraceMat(i,:)',10);
                    TraceMat(i,:) = Trace';
                end
            else
                for i = 1:size(TraceMat,1)
                    Trace = smooth(TraceMat(i,:)',varargin{1});
                    TraceMat(i,:) = Trace';
                end
            end
            obj.NeuTraceMat = TraceMat;
            obj.checkpoints{end+1} = 'smoothed';
        end

        function [obj] = func_calcAUC(obj)
            behlist = unique(obj.elab);
            for idx = behlist(2):behlist(end)
                labs(obj.elab~=idx) = 0;
                labs(obj.elab==idx) = 1;
                AUC = zeros(1,size(obj.NeuTraceMat,1));
                cellact = zeros(1,size(obj.NeuTraceMat,1));
                if length(obj.elab)>length(obj.NeuTraceMat)
                    labs = obj.elab(1,1:length(obj.NeuTraceMat));
                end

                for id = 1:size(obj.NeuTraceMat,1)
                    tic
                    [~,~,~,auc] = perfcurve(labs,obj.NeuTraceMat(id,:),1);
                    AUC(1,id) = auc;
                    cons = zeros(1,1000);
                    parfor JitterTime = 1:1000
                        I = randperm(length(obj.NeuTraceMat));
                        neumat = obj.NeuTraceMat(id,I);
                        [~,~,~,conauc] = perfcurve(labs,neumat,1);
                        cons(1,JitterTime) = conauc;
                    end
                    uplim = mean(cons)+3*std(cons);
                    lowlim = mean(cons)-3*std(cons);
                    if auc>uplim
                        cellact(1,id) = 1;
                    elseif auc<lowlim
                        cellact(1,id) = -1;
                    end
                    toc
                end
                obj.cellAUC{end+1} = AUC;
                obj.cellID{end+1} = cellact;
            end
            obj.checkpoints{end+1} = 'AUC calc';
        end

        function [sorted,I] = func_sortresp(obj,varargin)
            behlist = unique(obj.elab);
            for idx = behlist(2):behlist(end)
                mtrial = func_getTrialData(obj,idx,obj.params.preonset,obj.params.afteronset);
                %% rearrange by neuron
                AverPSTH = [];
                for ineuron = 1:size(mtrial{1},1)
                    nresA = [];
                    for tid = 1:length(mtrial)
                        nres = mtrial{tid}(ineuron,:);
                        nresA = [nresA; nres];
                    end
                    if length(mtrial)==1
                        Aver_res = nresA;
                    else
                        Aver_res = mean(nresA);
                    end
                    AverPSTH = [AverPSTH;Aver_res];
                end
                %% sort

                if ~isempty(varargin)
                    pre = varargin{1};
                    post = varargin{2};
                else
                    pre = obj.params.preonset;
                    post = obj.params.afteronset;
                end
                AverPSTH = func_CalcDeltaf(AverPSTH,obj.params.baseline(1),obj.params.baseline(2));
                [~,I] = sort(mean(AverPSTH(:,pre:post),2));
                sorted = (AverPSTH(I,:));

                %% plot
                figure
                imagesc(sorted)
                colorbar
                clim([-5 5])
                line([pre pre],[0 size(AverPSTH,1)+1],'Linestyle',':','Linewidth',2,'color','k')
                yticks(0:1:size(AverPSTH,1)+1)
                yticklabels(I)
            end
        end

        function [mtrial,tslidesM,obj] = func_getTrialData(obj,behidx,varargin)
            if isempty(varargin)
                preonset = 480; afteronset = 360;
            else
                preonset = varargin{1};
                afteronset = varargin{2};
            end
            binsize = 1;
            [SniMStart,SniMEnd,obj] = func_getStartEnd(obj,behidx);
            OnsetTraceA = {};
            tnum = 0;
            Mlen = length(SniMStart);
            for tid = 1:Mlen
                if tid == 1 || SniMStart(tid)>SniMEnd(tid-1)+90
                    try
                        OnsetAct = obj.NeuTraceMat(:,SniMStart(tid)-preonset:SniMStart(tid)+afteronset);
                        OnsetTraceA{end+1} = OnsetAct;
                        tnum = tnum + 1;
                    catch
                        continue
                    end
                end
            end
            mtrial = OnsetTraceA;
            time_xtrainM = {};
            for idx = 1:binsize:preonset+afteronset
                xtrain = [];
                for ii = 1:length(OnsetTraceA)
                    xii = OnsetTraceA{ii}(:,idx:idx+binsize);
                    xtrain = [xtrain mean(xii,2)];
                end
                time_xtrainM{end+1} = xtrain;
            end
            tslidesM = time_xtrainM;

        end

        function obj = get_trial(obj)
            behlist = unique(obj.elab);
            for idx = behlist(2):behlist(end)
                [mtrial,tslide,obj] = func_getTrialData(obj,idx,obj.params.preonset,obj.params.afteronset);
                obj.trials{end+1} = mtrial;
                obj.timeslides{end+1} = tslide;
            end
            obj.checkpoints{end+1} = 'Trial Got';
        end

        function obj = func_CalcDeltaf(obj,varargin)
            % zscore neutrace
            if isempty(varargin)
                %% Baseline deltaf/f
                MEAN = mean(obj.NeuTraceMat,2); Sigma = std(obj.NeuTraceMat,0,2);
                Normalized = (obj.NeuTraceMat - repmat(MEAN,1,size(obj.NeuTraceMat,2)))./repmat(Sigma,1,size(obj.NeuTraceMat,2));
            else
                TimeS = varargin{1}; TimeE = varargin{2};
                BaselineTime = [TimeS:TimeE];
                Baseline = obj.NeuTraceMat(:,BaselineTime); MEAN = mean(Baseline,2);Sigma = std(Baseline,0,2);
                Normalized = (obj.NeuTraceMat - repmat(MEAN,1,size(obj.NeuTraceMat,2)))./repmat(Sigma,1,size(obj.NeuTraceMat,2));
            end
            obj.Intermediate.NeuTracemat = obj.NeuTraceMat;
            obj.NeuTraceMat = Normalized;
            obj.checkpoints{end+1} = 'zscored';
        end

        function  PlotAllTrace(obj,varargin)
            % Plot the neuron traces and return the epochs of sniffs;
            xlen = [1:size(obj.NeuTraceMat,2)]/30;
            for N = 1:size(obj.NeuTraceMat,1)
                ylim(1) = max(obj.NeuTraceMat(N,:));
                ylim(2) = min(obj.NeuTraceMat(N,:));

                scrsz = get(0,'ScreenSize');
                figure1 = figure('Position',[0 30 scrsz(3) scrsz(4)-105]);
                plot(xlen,obj.NeuTraceMat(N,:),'linewidth',1)
                behlist = unique(obj.elab);
                hold on
                for ix = 2:length(behlist)
                    [SniMStart,SniMEnd] = func_getStartEnd(obj,behlist(ix));
                    SniMStart = SniMStart/30; SniMEnd = SniMEnd/30;
                    for j = 1:length(SniMStart)
                        hj = fill([SniMStart(j) SniMStart(j) SniMEnd(j) SniMEnd(j)],[ylim(2) ylim(1) ylim(1) ylim(2)],obj.colorlist{behlist(ix)});
                        set(hj,'edgealpha',0,'facealpha',0.2)
                    end
                    hold on
                end
                xticks(1:10:length(obj.NeuTraceMat)/30)
                if obj.save.saves
                    savepath = obj.save.savefig;
                    if isempty(dir(savepath))
                        mkdir(savepath)
                    end
                    print(gcf,'-r600','-dpng',[savepath,'\UnitCalcium',num2str(N),'.png']);
                    
                end
                close gcf

            end
        end

        function func_PlotCombineTrace(obj,varargin)
            Neu = obj.NeuTraceMat;
            scrsz = get(0,'ScreenSize');
            figure1 = figure('Position',[0 30 scrsz(3) scrsz(4)-105]);
            xlen = [1:size(Neu,2)]/30;
            for ii = 1:size(Neu,1)
                if ii > 1
                    w = (ii-1)*5;
                    plot(xlen,Neu(ii,:)+w)
                else
                    plot(xlen,Neu(ii,:))
                end
                hold on
            end
            behlist = unique(obj.elab);
            ylim(1) = max(Neu(ii,:)+w);
            ylim(2) = min(Neu(1,:));

            for ix = 2:length(behlist)
                [SniMStart,SniMEnd] = func_getStartEnd(obj,behlist(ix));
                SniMStart = SniMStart/30; SniMEnd = SniMEnd/30;
                for j = 1:length(SniMStart)
                    hj = fill([SniMStart(j) SniMStart(j) SniMEnd(j) SniMEnd(j)],[ylim(2) ylim(1) ylim(1) ylim(2)],obj.colorlist{behlist(ix)});
                    set(hj,'edgealpha',0,'facealpha',0.2)
                end
                hold on
            end
            xticks(1:20:length(Neu)/30)

            if obj.save.saves
                savepath = obj.save.savefig;
                print(gcf,'-r600','-dpng',[savepath,'\CombinedTrace.png']);
            end
            figure
            ylim(1) = 3;
            ylim(2) = -0.5;
            imagesc(Neu)
            colorbar
            hold on
            for ix = 2:length(behlist)
                [SniMStart,SniMEnd] = func_getStartEnd(obj,behlist(ix));
                for j = 1:length(SniMStart)
                    hj = fill([SniMStart(j) SniMStart(j) SniMEnd(j) SniMEnd(j)],-1*[ylim(2) ylim(1) ylim(1) ylim(2)],obj.colorlist{behlist(ix)});
                    set(hj,'edgealpha',0,'facealpha',0.2)
                end
                hold on
            end
            close gcf

        end

        function [Start,End,obj] = func_getStartEnd(obj,idx)
            elabx = obj.elab;
            elabx(elabx~=idx) = 0;
            elab1 = [0 elabx];
            EventP = [elabx 0] - elab1;
            Start = find(EventP==idx);
            End = find(EventP==-1*idx);
            obj.StartPoint{end+1} = Start;
            obj.EndPoint{end+1} = End;
            obj.checkpoints{end+1} = 'StartEndPoint Got';

        end

        function [elab1,obj] = func_getBeh(obj,BehaviorFile)
            % load behavior to generate elab
            try
                opt = detectImportOptions(BehaviorFile);
                Beh_f = readtable(BehaviorFile,opt,'ReadVariableNames',true);
                len = length(Beh_f.RecordingTime);
            catch
                opt = detectImportOptions(BehaviorFile);
                opt.VariableNames = {'TrialTime' 'RecordingTime' 'Subject' 'Behavior' 'Event'};
                Beh_f = readtable(BehaviorFile,opt,'ReadVariableNames',true);
                Beh_f = Beh_f(43:size(Beh_f,1),:);
                if strcmp(Beh_f.Event(1),'state stop')
                    Beh_f = readtable(BehaviorFile,opt,'ReadVariableNames',true);
                    Beh_f = Beh_f(42:size(Beh_f,1),:);
                end
            end
            if isempty(obj.beh)
                Beh_list = unique(Beh_f.Behavior);
                obj.beh = Beh_list;
            else
                Beh_list = obj.beh;
            end
            Behnum = length(Beh_list);
            for i = 1:Behnum
                disp(['The ',num2str(i),'th',' Behavior is ',Beh_list{i}])
            end
            len = length(Beh_f.RecordingTime);
            Start = [1:2:len-1]';
            RecTime = str2double(Beh_f.RecordingTime);
            if isnan(RecTime)
                RecTime = Beh_f.RecordingTime;
            end
            start_time = RecTime(Start);
            Stop = [2:2:len]';
            stop_time = RecTime(Stop);

            elab1 = zeros(1,18000);
            for i = 1:length(Start)
                timark = ceil(30*start_time(i)):floor(30*stop_time(i)); timark = timark';
                for idx = 1:Behnum
                    if strcmp(Beh_f.Behavior(Start(i)),Beh_list{idx})
                        elab1(timark) = idx;
                    end
                end
            end
            obj.elab = elab1;

        end

        function [aligned_elab,obj] = func_alignBeh(obj,dpath,animal,state,session)
            % align the behavior time stamp to miniscope time
            BehaviorFile = [dpath,animal,'\',state,'\Behavior',session,'\Export Files\','Beh.csv'];
            timestampFile = [dpath,animal,'\',state,'\Sess',session,'\behcam\timeStamps.csv'];
            msstampFile = [dpath,animal,'\',state,'\Sess',session,'\timeStamps.csv'];
            if  exist(BehaviorFile,'file')==2 && exist(timestampFile,'file')==2
                [~,obj] = func_getBeh(obj,BehaviorFile);
                elab_origin = obj.elab;
                behidx = unique(elab_origin);
                timestamp_beh = readtable(timestampFile);
                beh_tsmat = table2array(timestamp_beh);
                timestamp_ms = readtable(msstampFile);
                ms_tsmat = table2array(timestamp_ms);
                elabx = zeros(size(ms_tsmat(:,1)))';
                for id = 2:length(behidx)
                    [Start,End] = func_getStartEnd(obj,behidx(id));
                    [msloc,~] = find(beh_tsmat(:,1)==Start);
                    [meloc,~] = find(beh_tsmat(:,1)==End);

                    m_start = beh_tsmat(msloc,2);
                    m_end = beh_tsmat(meloc,2);

                    x_m_start = zeros(size(m_start)); x_m_end = x_m_start;

                    for iter = 1:length(m_start)
                        [x_m,~] = find(abs(ms_tsmat(:,2)-m_start(iter))<30); % a frame is around 30ms
                        x_m_start(iter) = x_m(1);
                        [x_m,~] = find(abs(ms_tsmat(:,2)-m_end(iter))<30);
                        x_m_end(iter) = x_m(1);
                    end


                    for iter = 1:length(x_m_start)
                        elabx(x_m_start(iter):x_m_end(iter)) = behidx(id);
                    end
                end
            elseif exist([dpath,animal,'\',state,'\Sess',session,'\timestamp.dat'],'file')==2
                [elab_ori,obj] = func_getBeh(obj,BehaviorFile);
                behidx = unique(elab_ori);
                tsdat = importdata([dpath,animal,'\',state,'\Sess',session,'\timestamp.dat']);
                timestamp_all = tsdat.data(3:end,:);
                camera_series = sort(unique(timestamp_all(:,1)));
                beh_tsmat = timestamp_all(find(timestamp_all(:,1)==camera_series(1)),2:3);
                ms_tsmat = timestamp_all(find(timestamp_all(:,1)==camera_series(2)),2:3);
                elabx = zeros(size(ms_tsmat(:,1)))';
                for id = 2:length(behidx)
                    [Start,End,obj] = func_getStartEnd(obj,behidx(id));
                    [msloc,~] = find(beh_tsmat(:,1)==Start);
                    [meloc,~] = find(beh_tsmat(:,1)==End);

                    m_start = beh_tsmat(msloc,2);
                    m_end = beh_tsmat(meloc,2);

                    x_m_start = zeros(size(m_start)); x_m_end = x_m_start;

                    for iter = 1:length(m_start)
                        [x_m,~] = find(abs(ms_tsmat(:,2)-m_start(iter))<30); % a frame is around 30ms
                        x_m_start(iter) = x_m(1);
                        [x_m,~] = find(abs(ms_tsmat(:,2)-m_end(iter))<30);
                        x_m_end(iter) = x_m(1);
                    end


                    for iter = 1:length(x_m_start)
                        elabx(x_m_start(iter):x_m_end(iter)) = behidx(id);
                    end
                end

            end
            try
                aligned_elab = elabx;
            catch
                disp('Check your behavior path as the program cannot detect related file')
            end

        end

    end

end

