% Program for  RBF..........................................
%kmeans.m file used along with this, is of octave package
clear all
close all
clc

% Load the training data..................................................
X=load('gcmtrain.dat');
[m,~] = size(X);
K = 100; 				%Number of Clusters
lam = 0.001;
lam2= 0.003;
lam3 = 0.001;
epo = 1000;

features = 98;
numClasses = 14;
y_train = X(:,features+1);
train_features = [];
x = [];
for i=1:features
    x = X(:,i);
    train_features = [train_features x];
end
perma_train = train_features;

A = X(:,99:112);
inpX=X(:,1:features);

centroids = zeros(K, size(train_features,2));
randid = randperm(size(train_features,1));
centroids = train_features(randid(1:K),:);
[id, centroids] = kmeans(train_features,K);

sigma = zeros(K,1);
ma=0;
for i=1:K
    for j=1:K
        sigmat=sum((centroids(i)-centroids(j)).^2);
        if(sigmat>ma)
            ma=sigmat;
       end
    end
end

for i=1:K
   sigma(i)=sqrt(ma);
end

process_train = [];
for i=1:K
    for j=1:features
       train_features(:,j) -= centroids(i,j);
    end
    process_train = [process_train train_features];
    train_features = perma_train;
end

for j=1:features*K
    process_train(:,j) = process_train(:,j).^2;
end
ans_train=[];
bleh = [];
w=1;
for j=1:features:features*K
    for q=1:features
        bleh = [bleh process_train(:,w)];
        w+=1;
    end
    bleh = sum(bleh,2);
    ans_train = [ans_train bleh];
    bleh = [];
end

for i=1:K
    ans_train(:,i) = -ans_train(:,i)/sigma(i)^2;
end
phi_train = exp(ans_train);

phi_train = [phi_train ones(m,1)];
weights = pinv(phi_train'*phi_train)*phi_train'*A;
%weights = 0.01*(rand(K,numClasses)*2.0 - 1.0);



%---------------------------Gradient descent
for ep = 1 : epo
    dwi=zeros(1,numClasses);
    dwi2=zeros(1,features);
    sigerr=0;
    for i = 1:m
        for l = 1:K
            diff = inpX(i,:)-centroids(l,:);
            diff = diff.^2;
            yphi(1,l) = exp(-sum(diff,2)/sigma(l)^2);
        end 
        yphi(1,K+1) = 1; 
              
        f_ans = yphi * weights;
        error = A(i,:) - yphi*weights;
        for sa = 1 : K
            dwi =  error.*yphi(1,sa);
            gradient = yphi(1,sa)*(error*weights(sa,:)')*(inpX(i,:)-centroids(sa,:)); %'
            gradient =  gradient/(sigma(sa)^2);
            temp = yphi(1,sa)*sum((inpX(i,:)-centroids(sa,:)).^2) * (error(1,:)*weights(sa,:)'); %'
            temp = temp/(sigma(sa)^3);
            sigma(sa) = sigma(sa) + lam3*(temp);
            centroids(sa,:)=centroids(sa,:)+ lam2*(gradient);
            weights(sa,:) = weights(sa,:)+ lam*dwi;
        end
        dwi = error.*yphi(1,K+1);
        weights(K+1,:) = weights(K+1,:) + lam*dwi;    
    end
end

%---------------------------------------''

Y=load('gcmtest.dat');
[n,~] = size(Y);
test_features = [];
y = [];
for i=1:features
    y = Y(:,i);
    test_features = [test_features y];
end
perma_test = test_features;
y_test = Y(:,features+1);
B = ones(numClasses,n)*-1;

for i=1:n
	B(y_test(i),i)=1;
end

process_test = [];
for i=1:K
    for j=1:features
       test_features(:,j) -= centroids(i,j);
    end
    process_test = [process_test test_features];
    test_features = perma_test;
end

for j=1:features*K
    process_test(:,j) = process_test(:,j).^2;
end

ans_test = [];
blah = [];
r=1;
for j=1:features:features*K
    for q=1:features
        blah = [blah process_test(:,r)];
        r+=1;
    end
    blah=sum(blah,2);
    ans_test = [ans_test blah];
    blah = [];
end

for i=1:K
    ans_test(:,i) = -ans_test(:,i)/sigma(i)^2;
end

phi_test = exp(ans_test);
phi_test = [phi_test ones(n,1)];
out = phi_test * weights;
[~,output] = max(out,[],2);

conftes = zeros(numClasses,numClasses);
correct_ones = 0;
%[output y_test]
for i=1:n
	conftes(output(i), y_test(i)) +=1;
end

for k=1:numClasses
	correct_ones = correct_ones + conftes(k,k);
end
correct_ones/n
conftes
