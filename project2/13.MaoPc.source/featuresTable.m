%% 
% Math Mini Project 2
% Count the occurences of the keyword in each paper
% Merge all the keywords from all papers into a big table with their count
%   input: nips dataset (1987-2016)
%   outputs: keywords in a 1xn cell, n=number of papers.
% ===================================================================

% get the unique keyword (feature) from all papers.
keyTable = table();
keyCell = cell(1,dataSize);
countCell = cell(1,dataSize);
for i = 1:length(keywords)
    key = categorical(keywords{i});
    key_data = categories(key);
    count_data = countcats(key);
%     keyI = strcat('k', num2str(i));
%     assignin('base',keyI,key_data);
%     countI = strcat('c', num2str(i));
%     assignin('base',countI,count_data);
    keyCell(i) = {key_data};
    countCell(i) = {count_data};
end

tempKey = union(keyCell{1}, keyCell{2});
for i = 3:length(keyCell)
    tempKey = union(tempKey, keyCell{i});
end

keyTable.key = tempKey;


% map the value to the key for each paper
cValue = [];
for i = 1:length(keyCell)
    cTemp = double(ismember(keyTable.key, keyCell{i}));
    cTemp(find(cTemp==1)) = countCell{i};
    cValue = [cValue, cTemp];
end