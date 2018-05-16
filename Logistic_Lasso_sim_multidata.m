clear;
clc;

%% Generating beta %%
beta=zeros(1,2000);
beta_int(1)=1.5;
beta_int(2)=-1.2;
beta_int(3)=1.8;
beta_int(4)=-2;
beta_int(5)=2.5;
beta_int(6)=-1.2;
beta_int(7)=1;
beta_int(8)=-1.5;
beta_int(9)=2;
beta_int(10)=-1.6;
beta_t=beta';

%% import training data
dat = importdata('data_combat.txt');
label = importdata('data_y.txt');
x_train = dat.data';
y_train = label;
train_size = size(x_train,1);

%% import testing data
dat1 = importdata('data_combat.txt');
label1 = importdata('data_y.txt');
x_test = dat1.data';
y_test = label1;
[test_size] = size(x_test,1);

%% Generating testing data
% test_size=200;
% x_test = normrnd(0, 1, test_size, size(beta,2));
% [n,p]=size(x_test);
% 
% % Setting correlation
% cor=0.2;             % correlation ¦Ñ=0.2, 0.4
% for i=1:n
%     for j=2:p
%         x_test(i,j) =  x_test(1,1) * cor + x_test(i,j) * (1-cor);
%     end
% end
% 
% sigm=0.2;            % noise sigm =0.2, 0.4, 0.6
% l = x_test * beta' + sigm * normrnd(0, 1, n, 1);
% prob=exp(l)./(1 + exp(l));
% for i=1:test_size
%     if prob(i)>0.5
%         y_test(i)=1;
%     else
%         y_test(i)=0;
%     end
% end
% y_test=y_test';

%%  Logistic + Lasso %%
col=size(x_train,2);
row=size(x_train,1);
beta=zeros(col,1);

%  calculating the beta_zero  %
temp=sum(y_train)/row;
beta_zero=log(temp/(1-temp));

% Inputting X, Y, beta_int and lambda %
beta_int=[beta_zero;beta];
beta_true=[beta_zero;beta_t];
x0=ones(row,1);
X=[x0,x_train];
Y=y_train;

% Setting lambda
lambda_max =norm(X'*Y,'inf'); % according to the https://github.com/yangziyi1990/SparseGDLibrary.git
lambda_min = lambda_max * 0.001;
m=10;
for i=1:m
    Lambda1(i)=lambda_max*(lambda_min/lambda_max)^(i/m);
    lambda=Lambda1(i);
    beta=Logistic_Lasso_func(X,Y,beta_int,lambda);   
    beta_path(:,i)=beta;
    fprintf('iteration times:%d\n',i);
end

[Opt,Mse]=CV_Lasso_logistic(X,Y,Lambda1);
beta_opt=beta_path(:,Opt);
fprintf('Beta: %f\n' ,beta_opt(1:11));

beta_zero=beta_opt(1); 
beta=beta_opt(2:end); 
l = beta_zero + x_test * beta;
prob=exp(l)./(1 + exp(l)); 
for i=1:test_size
    if prob(i)>0.5
        test_y(i)=1;
    else
        test_y(i)=0;
    end
end

error=test_y'-y_test;
error_number_testing=length(nonzeros(error))
beta_non_zero=length(nonzeros(beta_opt))

%% Performance
[accurancy,sensitivity,specificity]=performance(y_test,test_y');
fprintf('The accurancy of testing data (Lasso): %f\n' ,accurancy);
fprintf('The sensitivity of testing data (Lasso): %f\n' ,sensitivity);
fprintf('The specificity of testing data (Lasso): %f\n' ,specificity);

%% performance for training data
beta_zero=beta_opt(1); 
beta=beta_opt(2:end); 
l1 = beta_zero + x_train * beta;
prob1=exp(l1)./(1 + exp(l1)); 
for i=1:train_size
    if prob1(i)>0.5
        train_y(i)=1;
    else
        train_y(i)=0;
    end
end
error_train=train_y'-y_train;
error_number_train=length(nonzeros(error_train))

[accurancy_train,sensitivity_train,specificity_train]=performance(y_train,train_y');
fprintf('The accurancy of training data(Lasso): %f\n' ,accurancy_train);
fprintf('The sensitivity of training data (Lasso): %f\n' ,sensitivity_train);
fprintf('The specificity of training data (Lasso): %f\n' ,specificity_train);

%% performance for beta
[accurancy_beta,sensitivity_beta,specificity_beta]=performance_beta(beta_true,beta_opt);
fprintf('The accurancy of beta (Lasso): %f\n' ,accurancy_beta);
fprintf('The sensitivity of beta (Lasso): %f\n' ,sensitivity_beta);
fprintf('The specificity of beta (Lasso): %f\n' ,specificity_beta);

