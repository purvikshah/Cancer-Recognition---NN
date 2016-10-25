          function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm_npm(TrainingData_File, TestingData_File, Elm_Type, ActivationFunction)
          
          % Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
          % OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
          %
          % Input:
          % TrainingData_File     - Filename of training data set
          % TestingData_File      - Filename of testing data set
          % Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
          % NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
          % ActivationFunction    - Type of activation function:
          %                           'sig' for Sigmoidal function
          %                           'sin' for Sine function
          %                           'hardlim' for Hardlim function
          %                           'tribas' for Triangular basis function
          %                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
          %
          % Output: 
          % TrainingTime          - Time (seconds) spent on training ELM
          % TestingTime           - Time (seconds) spent on predicting ALL testing data
          % TrainingAccuracy      - Training accuracy: 
          %                           RMSE for regression or correct classification rate for classification
          % TestingAccuracy       - Testing accuracy: 
          %                           RMSE for regression or correct classification rate for classification
          %
          % MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
          % FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
          % neurons; neuron 5 has the highest output means input belongs to 5-th class
          %
          % Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
          % Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
          %
              %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
              %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
              %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
              %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
              %%%%    DATE:       APRIL 2004
          
          %%%%%%%%%%% Macro definition
          REGRESSION=0;
          CLASSIFIER=1;
          out=14;
          %%%%%%%%%%% Load training dataset
          train_data=load(TrainingData_File);
          T=train_data(:,end-out+1:end)';
          
          P=train_data(:,1:size(train_data,2)-out)';
          clear train_data;                                   %   Release raw training data array
          
          %%%%%%%%%%% Load testing dataset
          test_data=load(TestingData_File);
          %TV.T=test_data(:,end-13:end)';
          TV.T=test_data(:,end)';
          TV.P=test_data(:,1:size(test_data,2)-1)';
          clear test_data;                                    %   Release raw testing data array
          
          NumberofTrainingData=size(P,2);
          NumberofTestingData=size(TV.P,2);
          NumberofInputNeurons=size(P,1);
          
          %%%%%%%%%%%% Preprocessing the data of classification
              %sorted_target=sort(cat(2,T,TV.T),2);
              %label=zeros(1,1);                              %   Find and save in 'label' class label from training and testing data sets
              %label(1,1)=sorted_target(1,1);
              %j=1;
              %for i = 2:(NumberofTrainingData+NumberofTestingData)
               %   if sorted_target(1,i) ~= label(1,j)
                %      j=j+1;
                 %     label(1,j) = sorted_target(1,i);
                  %end
              %end
              %number_class=j;
              %NumberofOutputNeurons=number_class;
                 
              %%%%%%%%%% Processing the targets of training
              %temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
              %for i = 1:NumberofTrainingData
               %   for j = 1:number_class
                %      if label(1,j) == T(1,i)
                 %         break; 
                  %    end
                  %end
                  %temp_T(j,i)=1;
              %end
              %T=temp_T*2-1;
          
              %%%%%%%%%% Processing the targets of testing
              temp_TV_T=zeros(out, NumberofTestingData);
              for i = 1:NumberofTestingData
                  for j = 1:14
                      if j == TV.T(1,i)
                          break; 
                      endif
                  end
                  temp_TV_T(j,i)=1;
              end
              TV.T=temp_TV_T*2-1;
          
          
          max_avg_val_acc=0;
         % opti_InputWeight;
          %opti_BiasofHiddenNeurons;
          
          NumberofHiddenNeuronsArray = 5:5:50 % change to some other number 40
          %Code for 5x2 cross validation to find optimum parameters
          val_size = floor(NumberofTrainingData/5);
          val_size;
          for NumberofHiddenNeurons = NumberofHiddenNeuronsArray
          for i= 1:5
          InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
          BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
          
          avg_val_acc=0;
          avg_tra_acc=0;
          for j=1:2    % change to 2
          RandIndices = randperm(NumberofTrainingData);
          for i = 1:5  %change to 5
          
          if i==1
          valInd = RandIndices(1,1:val_size);
          trainInd = RandIndices(1,val_size + 1:end);
     
          
          elseif i==5
          valInd = RandIndices(1, 1+(i-1)*val_size: end);
          trainInd = RandIndices(1, 1:(i-1)*val_size);
          
          else
          valInd = RandIndices(1, 1+(i-1)*val_size:i*val_size);
          trainInd = [ RandIndices(1, 1:(i-1)*val_size) , RandIndices(1, i*val_size+1:end) ];
          endif
           %%%%%%%%%%% Calculate weights & biases
          %sstart_time_train=cputime;
          
          %%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
          
          tempP_tra= P(:, trainInd);
          
          tempT_tra= T(:,trainInd);
          tempH_tra=InputWeight*tempP_tra;
          %clear P;                                            %   Release input of training data 
          ind=ones(1,size(tempP_tra,2));
          BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
          tempH_tra=tempH_tra+BiasMatrix;
          
          %%%%%%%%%%% Calculate hidden neuron output matrix H
          switch lower(ActivationFunction)
              case {'sig','sigmoid'}
                  %%%%%%%% Sigmoid 
                  H_tra = 1 ./ (1 + exp(-tempH_tra));
              case {'sin','sine'}
                  %%%%%%%% Sine
                  H_tra = sin(tempH_tra);    
              case {'hardlim'}
                  %%%%%%%% Hard Limit
                  H_tra = double(hardlim(tempH_tra));
              case {'tribas'}
                  %%%%%%%% Triangular basis function
                  H_tra= tribas(tempH_tra);
              case {'radbas'}
                  %%%%%%%% Radial basis function
                  H_tra = radbas(tempH_tra);
                  %%%%%%%% More activation functions can be added here                
          end
          %clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
          
          %%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
          OutputWeight=pinv(H_tra') * tempT_tra';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
           
          
          %end_time_train=cputime;
          %TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM
          
          %%%%%%%%%%% Calculate the training accuracy
          Y_tra=(H_tra' * OutputWeight)';                             %   Y: the actual output of the training data
          
          clear H_tra;
          MissClassificationRate_Training=0;
              
              for i = 1 : size(tempT_tra, 2)
                  [x, label_index_expected]=max(tempT_tra(:,i));
                  [x, label_index_actual]=max(Y_tra(:,i));
                  if label_index_actual~=label_index_expected
                      MissClassificationRate_Training=MissClassificationRate_Training+1;
                  endif
              end
              TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
              avg_tra_acc=avg_tra_acc+TrainingAccuracy;
          
          %%%%% Validation Accuracy
          tempP_val=P(:, valInd);
          tempT_val=T(:, valInd);
          tempH_val=InputWeight*tempP_val;
          %clear TV.P;             %   Release input of testing data             
          ind=ones(1,size(tempP_val,2));
          BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
          tempH_val=tempH_val + BiasMatrix;
          switch lower(ActivationFunction)
              case {'sig','sigmoid'}
                  %%%%%%%% Sigmoid 
                  H_val = 1 ./ (1 + exp(-tempH_val));
              case {'sin','sine'}
                  %%%%%%%% Sine
                  H_val = sin(tempH_val);        
              case {'hardlim'}
                  %%%%%%%% Hard Limit
                  H_val = hardlim(tempH_val);        
              case {'tribas'}
                  %%%%%%%% Triangular basis function
                  H_val = tribas(tempH_val);        
              case {'radbas'}
                  %%%%%%%% Radial basis function
                  H_val = radbas(tempH_val);        
                  %%%%%%%% More activation functions can be added here        
          end
          VY=(H_val' * OutputWeight)';                 
          
          %%%%%%%%%% Calculate validation accuracy
              MissClassificationRate_Val=0;
              for i = 1 : size(tempT_val, 2)
                  [x, label_index_expected]=max(tempT_val(:,i));
                  [x, label_index_actual]=max(VY(:,i));
                  if label_index_actual~=label_index_expected
                      MissClassificationRate_Val=MissClassificationRate_Val+1;
                  endif
              end
              ValAccuracy=1-MissClassificationRate_Val/size(TV.T,2)
              avg_val_acc=avg_val_acc +ValAccuracy;    
           end
           avg_val_acc=avg_val_acc/5;
           avg_tra_acc=avg_tra_acc/5;
           end
           avg_val_acc=avg_val_acc/2;
           avg_tra_acc=avg_tra_acc/2;
           if avg_val_acc>max_avg_val_acc
           max_avg_val_acc=avg_val_acc;
           opti_no_neurons=NumberofHiddenNeurons;
           opti_InputWeight = InputWeight;
           opti_BiasofHiddenNeurons=BiasofHiddenNeurons;
           
           endif
           result=[NumberofHiddenNeurons avg_tra_acc avg_val_acc]       
          end
          
          end
          opti_no_neurons
          
  %%%%%%%%%%% Having found number of hidden neurons, find training accuracies and etsting accuracies
  start_time_train=cputime;
  
  %%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
  
  tempH=opti_InputWeight*P;
  size(opti_InputWeight)
  size(P)
  clear P;                                            %   Release input of training data 
  ind=ones(1,NumberofTrainingData);
  BiasMatrix=opti_BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
  tempH=tempH+BiasMatrix;
  
  %%%%%%%%%%% Calculate hidden neuron output matrix H
  switch lower(ActivationFunction)
      case {'sig','sigmoid'}
          %%%%%%%% Sigmoid 
          H = 1 ./ (1 + exp(-tempH));
      case {'sin','sine'}
          %%%%%%%% Sine
          H = sin(tempH);    
      case {'hardlim'}
          %%%%%%%% Hard Limit
          H = double(hardlim(tempH));
      case {'tribas'}
          %%%%%%%% Triangular basis function
          H = tribas(tempH);
      case {'radbas'}
          %%%%%%%% Radial basis function
          H = radbas(tempH);
          %%%%%%%% More activation functions can be added here                
  end
  clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
  
  %%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
  OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
   
  
  end_time_train=cputime;
  TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM
  
  %%%%%%%%%%% Calculate the training accuracy
  Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
  
  clear H;
          
          
          
          %%%%%%%%%%% Calculate the output of testing input
          %start_time_test=cputime;
          tempH_test=opti_InputWeight*TV.P;
          clear TV.P;             %   Release input of testing data             
          ind=ones(1,NumberofTestingData);
          BiasMatrix=opti_BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
          tempH_test=tempH_test + BiasMatrix;
          switch lower(ActivationFunction)
              case {'sig','sigmoid'}
                  %%%%%%%% Sigmoid 
                  H_test = 1 ./ (1 + exp(-tempH_test));
              case {'sin','sine'}
                  %%%%%%%% Sine
                  H_test = sin(tempH_test);        
              case {'hardlim'}
                  %%%%%%%% Hard Limit
                  H_test = hardlim(tempH_test);        
              case {'tribas'}
                  %%%%%%%% Triangular basis function
                  H_test = tribas(tempH_test);        
              case {'radbas'}
                  %%%%%%%% Radial basis function
                  H_test = radbas(tempH_test);        
                  %%%%%%%% More activation functions can be added here        
          end
          TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
          %end_time_test=cputime;
          %TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
        
        
        MissClassificationRate_Training=0;
        overall_train_Accuracy=0;
        average_train_acc = 0;
        geometric_train_acc = 1; 
        conftra=zeros(out,out);
        for i = 1 : size(T, 2)
            [x, label_index_expected]=max(T(:,i));
            [x, label_index_actual]=max(Y(:,i));
            if label_index_actual~=label_index_expected
                MissClassificationRate_Training=MissClassificationRate_Training+1;
            end
            conftra(label_index_expected,label_index_actual) = conftra(label_index_expected,label_index_actual)  + 1;

        end        
        display(conftra);
        
                for c = 1 : out
                  class_acc = conftra(c,c)/sum(conftra(c,:));
                  average_train_acc = average_train_acc + 100*class_acc;
                  geometric_train_acc = 100* geometric_train_acc* class_acc; 
                end
                geometric_train_acc= power(geometric_train_acc, 1/out)
                average_train_acc = average_train_acc/out
        MissClassificationRate_Training      
        overall_train_Accuracy=1-MissClassificationRate_Training/size(T,2)    
        
        MissClassificationRate_Testing=0;
        overall_test_Accuracy=0;
        average_test_acc = 0;
        geometric_test_acc = 1; 
        conftest=zeros(out,out);
        for i = 1 : size(TV.T, 2)
            [x, label_index_expected]=max(TV.T(:,i));
            [x, label_index_actual]=max(TY(:,i));
            if label_index_actual~=label_index_expected
                MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            end
            conftest(label_index_expected,label_index_actual) = conftest(label_index_expected,label_index_actual)  + 1;
        end
        display(conftest);
        
                for c = 1 : out
                  class_acc = conftest(c,c)/sum(conftest(c,:));
                  average_test_acc = average_test_acc + 100*class_acc;
                  geometric_test_acc = 100* geometric_test_acc* class_acc; 
                end
                geometric_test_acc= power(geometric_test_acc, 1/out)
                average_test_acc = average_test_acc/out
        MissClassificationRate_Testing
        overall_test_Accuracy=1-MissClassificationRate_Testing/size(TV.T,2)  
    
      
            
         