%% 
% Math Mini Project 2
% To categorize the data
% And extract some basic statistics
%   input: nips dataset (1987-2016)
%   outputs: keywords in a 1xn cell, n=number of papers.
% ===================================================================

% convert to categorical format
id = categorical(nips.id);
year = categorical(nips.year);
title = categorical(nips.title);
abstract = categorical(nips.abstract);

byYear = table();
byYear.year = categories(year);
byYear.count = countcats(year);

figure;
x = 1:1:height(byYear);
plot(x,byYear.count, '-b*', 'LineWidth', 4, 'MarkerSize', 15);
set(gca,'xticklabel',byYear.year);
xlabel('Year');
ylabel('Number of Published Paper');
set(gca,'FontSize',14);


