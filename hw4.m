clear; 
clc;
close all

load('SSVEP_Data.mat');


fstimulus(1)= 13; % we know frequency of stimulus 1 is 13 hz 
fstimulus(2)= 21; % we know frequency of stimulus 1 is 13 hz
fstimulus(3)= 17; % we know frequency of stimulus 1 is 13 hz

fs=256;% sampling rate is 256 hz
%% creating nice structure for dataset ssvep
data= cat(3,data1,data2,data3);
label= [ones(1,size(data1,3)), 2*ones(1,size(data2,3)),3*ones(1,size(data3,3))];
%%
slecHarmonic = 2;% first and second harmonic
channel_number = [1,8];

time=5; % 5 seconds tool keshideh baray recording
TimeAlltrails=time * 256;
t= linspace(0,time,TimeAlltrails);
Y1=[]; % first refrence
Y2=[]; % second refrence
Y3=[]; % third refrence
for n=1:slecHarmonic

    
    tp1(1,:)= sin(2*pi*(n*fstimulus(1))*t);% reference signal for fstimulus(1) is 13 hz
    tp1(2,:)= cos(2*pi*(n*fstimulus(1))*t);% reference signal for fstimulus(1) is 13 hz
    Y1=[Y1;tp1];

   
    tp2(1,:)= sin(2*pi*(n*fstimulus(2))*t);% reference signal for fstimulus(2) is 21 hz
    tp2(2,:)= cos(2*pi*(n*fstimulus(2))*t);% reference signal for fstimulus(2) is 21 hz
    Y2=[Y2;tp2];

   
    tp3(1,:)= sin(2*pi*(n*fstimulus(3))*t);% reference signal for fstimulus(3) is 17 hz
    tp3(2,:)= cos(2*pi*(n*fstimulus(3))*t);% reference signal for fstimulus(3) is 17 hz
    Y3=[Y3;tp3];
end


%%

[b,a]= butter(3,[49.5 50.5]/(fs/2), 'stop'); %  design notch filter for removing noise 50 hz 
%[b2 , a2] = butter(3 ,[10 100]/(fs/2), 'bandpass');
for i=1:480
    Signal= data(:,:,i);
    Signal= filtfilt(b,a,Signal);
    %Signal =filtfilt(b2 , a2,Signal);

% using CCA Function
r1 = myCCA(Signal(:,channel_number),Y1');
r2 = myCCA(Signal(:,channel_number),Y2');
r3 = myCCA(Signal(:,channel_number),Y3');

pearson_coefficient1 = max(r1);
pearson_coefficient2 = max(r2);
pearson_coefficient3 = max(r3);

Pearson_Coefficient = [pearson_coefficient1 , pearson_coefficient2 , pearson_coefficient3];
[MaxValue , index] = max(Pearson_Coefficient) ; 

numberStimulus(i) = index; % result : showing subject have attention which stimuluses

features(:,i) = Pearson_Coefficient';
end 

%% ACCuracy , performance
MatrixConfusion= confusionmat(label,numberStimulus);
m = diag(MatrixConfusion);
n = sum(MatrixConfusion(:)) ; 
Total_Acc= sum(m) /n  * 100;

disp(['Total accuracy using CCA: ', num2str(Total_Acc), ' %'])
 

%% ML model 

data1= features(:,label==1);
data2= features(:,label==2);
data3= features(:,label==3);

Ctotal = 0;
k = 5;%cross validation(k=5)
fold1= floor(size(data1,2) / k);
fold2= floor(size(data2,2) / k);
fold3= floor(size(data3,2) / k);

for i=1:k

    indxtest1= (i-1)*fold1+1:i*fold1;
    indxtrain1=1:size(data1,2);
    indxtrain1(indxtest1)=[];
    
    indxtest2= (i-1)*fold2+1:i*fold2;
    indxtrain2=1:size(data2,2);
    indxtrain2(indxtest2)=[];
    
        
    indxtest3= (i-1)*fold3+1:i*fold3;
    indxtrain3=1:size(data3,2);
    indxtrain3(indxtest3)=[];
    
    traindata1= data1(:,indxtrain1);
    testdata1= data1(:,indxtest1);
    %
    traindata2= data2(:,indxtrain2);
    testdata2= data2(:,indxtest2);
    
    traindata3= data3(:,indxtrain3);
    testdata3= data3(:,indxtest3);
  % concatination 
traindata= [traindata1,traindata2,traindata3];
trainlabel= [ones(1,size(traindata1,2)),2*ones(1,size(traindata2,2)),3*ones(1,size(traindata3,2))];
       
testdata= [testdata1,testdata2,testdata3];
testlabel= [ones(1,size(testdata1,2)),2*ones(1,size(testdata2,2)),3*ones(1,size(testdata3,2))];
 
% fit knn model 
model= fitcknn(traindata',trainlabel,'NumNeighbors',5); % number of neighbors is 5
output= predict(model,testdata')';
 
    ConfusionMatrix= confusionmat(testlabel,output);
    Ctotal= Ctotal+ ConfusionMatrix;
    accuracy(i)= sum(diag( ConfusionMatrix)) / sum( ConfusionMatrix(:))*100;
    accuracy1(i)=  ConfusionMatrix(1,1) / sum( ConfusionMatrix(1,:))*100;
    accuracy2(i)=  ConfusionMatrix(2,2) / sum( ConfusionMatrix(2,:))*100;
    accuracy3(i)=  ConfusionMatrix(3,3) / sum( ConfusionMatrix(3,:))*100;

end
Ctotal;
disp(['Total Accuracy using KNN(k=5): ',num2str(mean(accuracy)),'%'])
disp(['Accuracy for class 1: ',num2str(mean(accuracy1)),'%'])
disp(['Accuracy for class 2: ',num2str(mean(accuracy2)),'%'])
disp(['Accuracy for class 3: ',num2str(mean(accuracy3)),'%'])


%%
figure
%plot3(features(1,:),features(2,:),features(3,:),'Og')
plot3(features(1,label ==1),features(2,label == 1),features(3,label == 1),'Og');
hold on 
plot3(features(1,label ==2),features(2,label == 2),features(3,label == 2),'Ob');
hold on
plot3(features(1,label ==3),features(2,label == 3),features(3,label == 3),'Or')

%%
%%classificationLearner
save ('myssvep.mat' ,'label','features');

%%
load('myssvep.mat')
data1= features(:,label==1);
data2= features(:,label==2);
data3= features(:,label==3);
clearvars -except data1 data2 data3


k = 5 ;%cross validation when set k_fold is 5
fold1= floor(size(data1,2) / k);
fold2= floor(size(data2,2) / k);
fold3= floor(size(data3,2) / k);
Ctotal=0;
for i=1:k
    indxtest1= (i-1)*fold1+1:i*fold1;
    indxtrain1=1:size(data1,2);
    indxtrain1(indxtest1)=[];
    
    indxtest2= (i-1)*fold2+1:i*fold2;
    indxtrain2=1:size(data2,2);
    indxtrain2(indxtest2)=[];
    
        
    indxtest3= (i-1)*fold3+1:i*fold3;
    indxtrain3=1:size(data3,2);
    indxtrain3(indxtest3)=[];
    
    traindata1= data1(:,indxtrain1);
    testdata1= data1(:,indxtest1);
    %
    traindata2= data2(:,indxtrain2);
    testdata2= data2(:,indxtest2);
    
    traindata3= data3(:,indxtrain3);
    testdata3= data3(:,indxtest3);
    %
    traindata= [traindata1,traindata2,traindata3];
    trainlabel= [ones(1,size(traindata1,2)),2*ones(1,size(traindata2,2)), 3*ones(1,size(traindata3,2))];
       
    
    testdata= [testdata1,testdata2,testdata3];
    testlabel= [ones(1,size(testdata1,2)),2*ones(1,size(testdata2,2)),3*ones(1,size(testdata3,2))];
        
   
   
    SVMModel =multisvmtrainOvO(traindata,trainlabel,'rbf');
    output=multiclassSVMOvO(SVMModel,testdata);

    ConfusionMatrix= confusionmat(testlabel,output);
    Ctotal= Ctotal+ ConfusionMatrix;

    accuracy(i)= sum(diag( ConfusionMatrix)) / sum( ConfusionMatrix(:))*100;
    accuracy1(i)=  ConfusionMatrix(1,1) / sum( ConfusionMatrix(1,:))*100;
    accuracy2(i)=  ConfusionMatrix(2,2) / sum( ConfusionMatrix(2,:))*100;
    accuracy3(i)=  ConfusionMatrix(3,3) / sum( ConfusionMatrix(3,:))*100;

end
Ctotal

disp(['Total Acc using SVM is: ',num2str(mean(accuracy)),'%'])
disp(['Acc for class 1 is: ',num2str(mean(accuracy1)),'%'])
disp(['Acc for class 2 is : ',num2str(mean(accuracy2)),'%'])
disp(['Acc for class 3 is: ',num2str(mean(accuracy3)),'%'])
