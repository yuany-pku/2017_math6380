%% 
% Math Mini Project 2
% To fix the abstract missing problem
% Extract the keyword from both title and abstract after fixing
%   input: nips dataset (1987-2016)
%   outputs: keywords in a 1xn cell, n=number of papers.
% ===================================================================

%% Rearrange the data by year.
nips = sortrows(nips, 'id');
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
    startIndex = strmatch(abstToMatch, abstractCell);
    if isempty(startIndex)
        startIndex = length(strsplit(char(nips.title(mIndex(j))), ...
            unwantedStr)) + 2;
    end
    
    introToMatch = {'introduction'};
    endIndex = strmatch(introToMatch, abstractCell);
    if isempty(endIndex)
        endIndex = startIndex +200;
    end
   
    keywords(mIndex(j)) = {abstractCell'};
end

% case 2: when abstract is available: join title and abstract
nIndex = find(~strcmp(nips.abstract, 'Abstract Missing'));
ta = strcat(nips.title(nIndex), {' '}, nips.abstract(nIndex));
for i=1:length(nIndex)
    keywords(:,nIndex(i)) = {strsplit(lower(char(ta(i))), unwantedStr)'};
end


% further delete meaningless keywords
for k = 1: length(keywords)
    for alphabet = 'a':'z'
        ind = find(strcmp(string(keywords{k}),alphabet));
        keywords{k}(ind) = [];
    end
    keywords{k}(find(strcmp(string(keywords{k}),''))) = [];
end

