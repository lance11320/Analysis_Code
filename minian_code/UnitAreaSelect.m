clear
close all
animal = { '882'};
state = { 'Diestrus'};% 
session = { '3'};
%session = {'1'};
warning off
for i = 1:length(animal)
    for j = 1:length(state)
        for k = 1:length(session)
            dpath = ['K:\MJH\SortMS\M',animal{i},'\',state{j},'\Sess',session{k}];
            if isempty(dir(dpath))
                break
            else
                Arr = load([dpath,'Res\',animal{i},'A.mat']);
                area = Arr.array;
                map = squeeze(sum(area,1));
                figure
                imshow(map)
                bwmap = im2bw(map,0.2); % use 0.2 as a threshold to get clear boundary
                B = cell(size(area,1),1);
                centroid = zeros(size(area,1),2);
                for umap_id = 1:size(area,1)
                    single_map = squeeze(area(umap_id,:,:));
                    bwmap = im2bw(single_map,0.2);
                    [b,L] = bwboundaries(bwmap,'noholes');
                    stats = regionprops(L,'Area','Centroid');
                    B{umap_id} = b{1};
                    cent = stats(1).Centroid;
                    centroid(umap_id,:) = cent;
                end
                
                
                figure
                for UId = 1:length(B)
                    boundary = B{UId};
                    
                    plot(boundary(:,2),size(bwmap,2)-boundary(:,1),'k','LineWidth',1)
                    %fill(boundary(:,2),size(bwmap,2)-boundary(:,1),'r')
                    text(centroid(UId,1),size(bwmap,2)-centroid(UId,2),num2str(UId),'FontSize',14,'Color','r')
                    hold on
                    

                end
                try
                [x,y] = getpts;% get chose points, choose the desired units and press 'enter'
                y = size(bwmap,2) - y;
                ChoseUnit = [];
                for xall = 1:length(x)
                    xi = x(xall);
                    yi = y(xall);
                    distance = sqrt((xi-centroid(:,1)).^2+(yi-centroid(:,2)).^2);
                    single_unit_id = find(distance == min(distance));
                    ChoseUnit = [ChoseUnit single_unit_id]
                end
                %diag(map(floor(y),floor(x)))
                UnitChosen = setdiff(1:size(area,1),ChoseUnit);
                newarea = area(UnitChosen,:,:);
                save([dpath,'Res\',animal{i},'Chose_A.mat'],'newarea')
                catch
                    disp('No Modification Here')
                end




            end
        end
    end
end
