clear
close all

home = 'J:\MJH\';
root = 'SortMS\M';
PETHhome = 'K:\MJH\MSRes\M' ;
% animal = {'893' '882' '886' '955' '959' '983'};
% state = {'Diestrus' };
% sess = { {'1' '4'}  {'3','4'} {'1', '2'} {'2' '3'} {'1'} {'1'}} ;
% animal = {'900' '910' '918' '919' '970' '65' '82' '90' '91' '92'};
% state = {'Male' };
% sess = { {'5','9','10'} {'2','5','6'} {'1'} {'2'}  {'1' '2' '3'} {'1' '2'} {'1'} {'1'} {'1'} {'1'}} ;
% animal = {'342' '352' '451' '882' '883' '954' '955' '959' '983' };
% state = {'Estrus' };
% sess = { {'1','2'} {'1', '2'} {'1'} {'1','4'} {'1'} {'1'} {'1'} {'1' '2'} {'1'}} ; 
animal = {'342'};
state = {'Estrus'};
sess = {{'1'}};

for i = 1:length(animal)
    for j = 1:length(state)
        session = sess{i};
        for k = 1:length(session)
            dpath = [home,root,animal{i},'\',state{j},'\Sess',session{k}];
            PETHpath = [PETHhome,animal{i},'\',state{j},'\Sess',session{k},'Res\PETH.mat'];
            if exist(dpath,'dir')==0
                continue
            else
                obj = PETHdata(PETHpath,home,root,animal{i},state{j},session{k});
                obj.colorlist = {'r','b','g','y'};% each color is for a behavior respectively
                obj.params.preonset = 120; obj.params.afteronset = 120; % frames relative to behavior onset
                obj.params.baseline = [1 60]; % baseline for trial heatmap
                obj.save.saves = false;% save your plot fig or not
                obj.beh = {}; % please asign the value if you have known your behavior list

                obj = obj.func_CalcDeltaf();
                obj = obj.matsmooth(20);
                %obj.PlotAllTrace();
                %obj.func_PlotCombineTrace();
                %obj = obj.func_calcAUC();
                obj = obj.get_trial();


            end

        end
    end
end