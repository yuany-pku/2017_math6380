
close all; clear all; clc;
load snp452-data.mat
R=log(X(2:end,:)./X(1:end-1,:));
[m,n]=size(R);
% process the raw data
Radjusted=R; 
for k=1:n
for j=2:m-1
    if (abs(R(j,k)-R(j-1,k))> 0.4 && abs(R(j,k)-R(j+1,k))> 0.4) || abs(R(j,k))>0.4
    Radjusted(j,k)=1/2*(R(j-1,k)+R(j+1,k));
    end
end
if abs(R(m,k)-R(m-1,k))>0.4 || abs(R(m,k))>0.4
    Radjusted(m,k)=R(m-1,k);
end
end

% stocks within ten sectors 
Index1=[]; Index2=[]; Index3=[]; Index4=[]; Index5=[]; 
Index6=[]; Index7=[]; Index8=[]; Index9=[]; Index10=[]; 
code8={}; n8=0; % names within the energy sector 
for k=1:452
cc = stock{k}.class;
switch cc
    case '"Industrials"'
         Index1=[Index1,k];
    case char('"Financials"')     
         Index2=[Index2,k];
    case char('"Health Care"')   
         Index3=[Index3,k];
    case char('"Consumer Discretionary"')   
         Index4=[Index4,k];
    case char('"Information Technology"')   
         Index5=[Index5,k];  
    case char('"Utilities"')   
         Index6=[Index6,k];       
    case char('"Materials"')   
         Index7=[Index7,k];
    case char('"Energy"')   
         Index8=[Index8,k]; 
         n8=n8+1;
         code8{n8}=stock{k}.code;
%          char(stock{k}.code)
%          code8=[code8; char(stock{k}.code)];
    case char('"Telecommunications Services"')   
         Index9=[Index9,k];   
    otherwise 
         Index10=[Index10,k];      
end
end


R1=Radjusted(:,Index1'); R2 =Radjusted(:,Index2'); R3 =Radjusted(:,Index3'); ...
    R4 =Radjusted(:,Index4'); R5 =Radjusted(:,Index5');  
R6=Radjusted(:,Index6'); R7 =Radjusted(:,Index7'); R8 =Radjusted(:,Index8'); ...
    R9 =Radjusted(:,Index9'); R10 =Radjusted(:,Index10');  

figure()
boxplot(R8,'orientation','horizontal','labels',code8)

X1=X(:,Index1'); X2 =X(:,Index2'); X3 =X(:,Index3'); ...
    X4 =Radjusted(:,Index4'); X5 =X(:,Index5');  
X6=X(:,Index6'); X7 =X(:,Index7'); X8 =X(:,Index8'); ...
    X9 =X(:,Index9'); X10 =X(:,Index10');  

%---plot correlation matrix 
figure
colorbar
subplot(2,5,1);
C1=corr(R1);imagesc(C1);%colorbar;
title('Industrials')
subplot(2,5,2);
C2=corr(R2);imagesc(C2);%colorbar;
title('Financials')
subplot(2,5,3);
C3=corr(R3);imagesc(C3);%colorbar
title('HealthCare')
subplot(2,5,4);
C4=corr(R4);imagesc(C4);%colorbar
title('Con-Disc')
subplot(2,5,5);
C5=corr(R5);imagesc(C5);%colorbar
title('Inf-Tech')
subplot(2,5,6);
C6=corr(R6);imagesc(C6);%colorbar
title('Utilities')
subplot(2,5,7);
C7=corr(R7);imagesc(C7);%colorbar
title('Materials')
subplot(2,5,8);
C8=corr(R8);imagesc(C8);%colorbar
title('Energy')
subplot(2,5,9);
C9=corr(R9);imagesc(C9);%colorbar
title('Tel-Serv')
subplot(2,5,10);
C10=corr(R10);imagesc(C10);
title('Con-Stap')
set(gca,'FontSize',16)
set(gcf, 'PaperPosition', [0 0 5.90 4.9]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.90 4.9]); %Set the paper t
 saveas(gcf, 'corrplot.pdf'); close all; 

% PCA 
[wcoeff2,score2,latent2,tsquared2,explained2] = pca(R2,...
'VariableWeights','variance');figure()
pareto(explained2)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'explainedsec2.pdf'); close all; 



[wcoeff6,score6,latent6,tsquared6,explained6] = pca(R6,...
'VariableWeights','variance');figure()
pareto(explained6)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'explainedsec6.pdf');close all; 

[wcoeff7,score7,latent7,tsquared7,explained7] = pca(R7,...
'VariableWeights','variance');figure()
pareto(explained7)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'explainedsec7.pdf'); close all; 

[wcoeff8,score8,latent8,tsquared8,explained8] = pca(R8,...
'VariableWeights','variance'); 
pareto(explained8)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'explainedsec8.pdf'); close all; 
close all; 

% distance to center 
[st2,index2] = sort(tsquared2,'descend'); % sort in descending order
extreme2 = index2(1:30);
[st2,index6] = sort(tsquared6,'descend'); % sort in descending order
extreme6 = index6(1:30);
[st2,index7] = sort(tsquared7,'descend'); % sort in descending order
extreme7 = index7(1:30);
[st2,index8] = sort(tsquared8,'descend'); % sort in descending order
extreme8 = index8(1:30);
[extreme2 extreme6 extreme7 extreme8];

% the explain variance plot for Energy 
figure()
pareto(explained8)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'explainedsec8.pdf'); close all; 

% MDS chebychev
D = pdist(R8','chebychev');
[Y,eigvals] = cmdscale(D);
[eigvals eigvals/max(abs(eigvals))];
% eigenvalue plot 
plot(1:25,eigvals(1:25),'bo-');
line([1,length(eigvals)],[0 0],'LineStyle',':','XLimInclude','off',...
     'Color',[.7 .7 .7])
axis([1,length(eigvals),min(eigvals),max(eigvals)*1.1]);
xlabel('Eigenvalue number');
ylabel('Eigenvalue');
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'evalues.pdf');

% location map 
plot(Y(:,1), Y(:,2), '.'); 
% gname 
code=[4, 13,30, 35, 37];
text(Y(idnew,1),Y(idnew,2),code8(idnew),'fontsize',14)
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'cmddistance.pdf');


D = pdist(R8','correlation');
[Y,eigvals] = cmdscale(D);
[eigvals eigvals/max(abs(eigvals))];
% eigenvalue plot 
plot(1:25,eigvals(1:25),'bo-');
line([1,length(eigvals)],[0 0],'LineStyle',':','XLimInclude','off',...
     'Color',[.7 .7 .7])
axis([1,length(eigvals),min(eigvals),max(eigvals)*1.1]);
xlabel('Eigenvalue number');
ylabel('Eigenvalue');
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'evalues2.pdf');

% location map 
plot(Y(:,1), Y(:,2), '.'); 
%gname % mark points
idnew=[9 34 35 37 36];
text(Y(idnew,1),Y(idnew,2),code8(idnew),'fontsize',14)
set(gcf, 'PaperPosition', [0 0 5.90 4.9]);
set(gcf, 'PaperSize', [5.90 4.9]);
set(gca,'FontSize',16)
saveas(gcf, 'cmddistance2.pdf');




         
         