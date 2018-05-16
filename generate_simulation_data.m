clear;
clc;

%% Generating beta
beta=zeros(1,2000);
beta(1)=1;
beta(2)=-1;
beta(3)=1;
beta(4)=-1;
beta(5)=1;
beta(6)=-1;
beta(7)=1;
beta(8)=-1;
beta(9)=1;
beta(10)=-1;
beta_t=beta';

%% Generating simulation data 
train_size=200;
X = normrnd(0, 1, train_size, size(beta,2));
[n,p]=size(X);

% Setting correlation
cor=0.2;             % correlation ¦Ñ=0.2, 0.4
for i=1:n
    for j=2:p
        X(i,j) =  X(1,1) * cor + X(i,j) * (1-cor);
    end
end

sigm = 0.3;
l = X * beta' + sigm * normrnd(0, 1, n, 1);
prob=exp(l)./(1 + exp(l));

for i=1:train_size
    if prob(i)>0.5
        Y(i)=1;
    else
        Y(i)=0;
    end
end

% combine data
X=X';
Data=[Y;X];

index=zeros(p+1,1);
for i=1:p+1
    index(i)=i-1;
end

Dataset=[index,Data];
dlmwrite('D:\Ziyi\School\PMO\metanalysis\Simulation\combine\data\Dataset3_series_matrix_02.txt',Dataset,"\t");
    
