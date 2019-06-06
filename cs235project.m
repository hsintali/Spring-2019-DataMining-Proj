clc;
clear;
close;

%% download data
data = readtable('bank-additional-full.csv');

age = double(data.age);
duration = double(data.duration);
campaign = double(data.campaign);
pdays = double(data.pdays);
previous = double(data.previous);
emp_var_rate = double(data.emp_var_rate);
cons_price_idx = double(data.cons_price_idx);
cons_conf_idx = double(data.cons_conf_idx);
euribor3m = double(data.euribor3m);
nr_employed = double(data.nr_employed);

  subplot(3,3,1);histogram(data.age)
  xlabel('City index')
  xlabel('age')
  
  subplot(3,3,2);histogram(data.duration)
  xlabel('City index')
  xlabel('duration')
  
    subplot(3,3,3);histogram(data.campaign)
  xlabel('City index')
  xlabel('campaign')

    subplot(3,3,4);histogram(data.pdays)
  xlabel('City index')
  xlabel('pdays')

    subplot(3,3,5);histogram(data.previous)
  xlabel('City index')
  xlabel('previous')

    subplot(3,3,6);histogram(data.previous)
  xlabel('City index')
  xlabel('previous')
  
    subplot(3,3,7);histogram(data.emp_var_rate)
  xlabel('City index')
  xlabel('emp_var_rate')
  
    subplot(3,3,8);histogram(data.cons_price_idx)
  xlabel('City index')
  xlabel('cons_price_idx')
    subplot(3,3,6);histogram(data.cons_conf_idx)
  xlabel('City index')
  xlabel('cons_conf_idx')
    subplot(3,3,9);histogram(data.euribor3m)
  xlabel('City index')
  xlabel('euribor3m')
  
  %% coverting data
catnames = {'no','yes'};
valueset = {'0','1'};
result = categorical(data.y,catnames,valueset);
result = double(result)-1;

catnames = {'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'};
valueset = {'0','1','2','3','4','5','6','7','8','9','10','11'};
job = categorical(data.job,catnames,valueset);
job = double(job)-1;

catnames = {'divorced','married','single','unknown'}; %note: 'divorced' means divorced or widowed)
valueset = {'0','1','2','3'};
marital = categorical(data.marital,catnames,valueset);
marital = double(marital)-1;

catnames = {'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'};
valueset = {'0','1','2','3','4','5','6','7'};
education = categorical(data.education,catnames,valueset);
education = double(education)-1;

catnames = {'no','yes','unknown'};
valueset = {'0','1','3'};
default = categorical(data.default,catnames,valueset);
default = double(default)-1;

catnames = {'no','yes','unknown'};
valueset = {'0','1','3'};
housing = categorical(data.housing,catnames,valueset);
housing = double(housing)-1;

catnames = {'no','yes','unknown'};
valueset = {'0','1','2'};
loan = categorical(data.loan,catnames,valueset);
loan = double(loan)-1;

catnames = {'cellular','telephone'};
valueset = {'0','1'};
contact = categorical(data.contact,catnames,valueset);
contact = double(contact)-1;

catnames = { 'jan', 'feb', 'mar','apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'};
valueset = {'0','1','2','3','4','5','6','7','8','9','10','11'};
month = categorical(data.month,catnames,valueset);
month = double(month)-1;

catnames = {'mon','tue','wed','thu','fri'};
valueset = {'0','1','2','3','4'};
day_of_week = categorical(data.day_of_week,catnames,valueset);
day_of_week = double(day_of_week)-1;

catnames = {'failure','nonexistent','success'};
valueset = {'0','1','2'};
poutcome = categorical(data.poutcome,catnames,valueset);
poutcome = double(poutcome)-1;
%  X = zeros(4119,20);
X1 = [age,job,marital,education,default,housing,loan,contact,month, day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed];
[coeff,score,latent,tsquared] = pca(X1);
figure;
percent_explained= 100*latent/sum(latent); %cumsum(latent)./sum(latent)
pareto(percent_explained);% https://blog.csdn.net/ckzhb/article/details/75281727
xlabel('PrincipalComponent');
ylabel('VarianceExplained (%)');

[idx,C,sumd,D] = kmeans(coeff,10);

datanorm = [X1,result];
datanorm = mapminmax(datanorm',0,1);
datanorm = datanorm';

%https://blog.csdn.net/watkinsong/article/details/8234766
%   valueset = 1:3;
% catnames = {'red' 'green' 'blue'};
% 
% A = categorical([1 3; 2 1; 3 1],valueset,catnames);