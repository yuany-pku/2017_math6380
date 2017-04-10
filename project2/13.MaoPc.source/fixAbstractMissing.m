%% 
% Math Mini Project 2
% To fix the abstract missing problem
% Extract the keyword from both title and abstract after fixing
% input: nips dataset (1987-2016)
% outputs: 
% ===================================================================

%%
dataSize = height(nips);
keywords = cell(1,dataSize);

% Exclude the following characters / noise
unwantedStr = {'.', ',', ' ', '\n', '?', '''', '<', '>','"', '*', '~', ...
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '(', ')', ';', '+', ...
    '-', '=', '   ', '!', '#', '  ', '    ', '%', '|', '&', '\', '[', ']', ...
    '...', '^', '\tab', '$', '/', '{', '}' ':', '_', '!!!', '!!', '      ',...
    '`', '``', '@', '', '', '	', '', '', '', ''}; 

% case 1: solve the abstract missing problem
mIndex = find(strcmp(nips.abstract, 'Abstract Missing'));
abstractM = nips.paper_text(mIndex);
for j=1:length(mIndex)
    startIndex = []; endIndex = [];
    abstractCell = lower(strsplit(char(abstractM(j)), unwantedStr));
    abstToMatch = {'abstract'};
    k = 1;
    while isempty(startIndex)
        startIndex = strmatch(abstToMatch(k), abstractCell);
        k = k + 1;
        if k == 4
            startIndex = length(strsplit(char(nips.title(mIndex(j))), ...
                unwantedStr)) + 2;
            break;
        end
    end
    introToMatch = {'introduction'};
    k = 1;
    while isempty(endIndex) 
        endIndex = strmatch(introToMatch(k), abstractCell);
        k = k + 1;
        if k == 4
            endIndex = startIndex +200;
            break;
        end
    end    
    tb = strcat(nips.title(mIndex(j)), {' '}, ...
        strjoin(abstractCell(startIndex+1:endIndex-1)));
    % nips.abstract(mIndex(j)) = strjoin(abstractCell(startIndex+1:endIndex-1));
    keywords(mIndex(j)) = {strsplit(lower(char(tb)), unwantedStr)'};
end

% case 2: when abstract is available: join title and abstract
nIndex = find(~strcmp(nips.abstract, 'Abstract Missing'));
ta = strcat(nips.title(nIndex), {' '}, nips.abstract(nIndex));
for i=1:length(nIndex)
    keywords(:,nIndex(i)) = {strsplit(lower(char(ta(i))), unwantedStr)'};
end




