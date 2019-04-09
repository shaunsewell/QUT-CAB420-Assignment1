%% CAB420 Assignment 1 
%   Shaun Sewell N9509623 
%
%% 1. Features, Classes, and Linear Regression
clear ; close all; clc
% (a) Plot the training data in a scatter plot.

% Load training data and separate features
mTrain = load('data/mTrainData.txt'); 
Xtr = mTrain(: ,1); Ytr = mTrain(: ,2);

% Plot training data
figure('name', 'Scatter Plot of Training Data');
plot (Xtr, Ytr, 'bo'); 
xlabel('X');
ylabel('Y');
title('Scatter Plot of Training Data');

% (b) Create a linear regression learner using the above functions. Plot it on
%the same plot as the training data.

Xtr_2 =  polyx(Xtr,2);
% train a regression learner
regress_learner = linearReg(Xtr_2 ,Ytr); 
xline = [0:.01:2]' ; % transpose : make a column vector , like training x
yline = predict( regress_learner , polyx (xline ,2) );

% Plot predictions
figure('name', 'Quadratic linear regression predictor');
plot(xline, yline, 'ro');
% Plot training data
hold on
plot (Xtr, Ytr, 'bo');

% Set figure properties
xlim([0 1]);
ylim([0 4.5]);
xlabel('X');
ylabel('Y');
legend('Linear Predictor', 'Training Data');
title('Quadratic linear regression predictor');

% (c) Create plots with the data and a higher-order polynomial (3, 5, 7, 9, 11, 13)

Xtr_7 =  polyx(Xtr,7);
% train the learner
septic_learner = linearReg(Xtr_7 ,Ytr);                   
xline = [0:.01:2]' ;                                % transpose : make a column vector , like training x
yline = predict( septic_learner , polyx (xline ,7) );

% Plot predictions
figure('name', "Septic linear regression predictor");
plot(xline, yline, 'ro');
hold on 
% Plot training data
plot (Xtr, Ytr, 'bo');
% Set figure properties
xlim([0 1]);
ylim([0 4.5]);
xlabel('X');
ylabel('Y');
legend('Linear Predictor', 'Training Data');
title('Septic linear regression predictor');

% (d) Calculate the mean squared error (MSE) associated with each of your
% learned models on the training data.

% Quadratic model
MSE_Quadratic_Trained = mse(regress_learner,Xtr_2, Ytr);
fprintf('The MSE for the quadratic predictor on training data was: %.2f\n', MSE_Quadratic_Trained);

% Septic model
MSE_Septic_Trained = mse(septic_learner,Xtr_7, Ytr);
fprintf('The MSE for the septic predictor on training data was: %.2f\n', MSE_Septic_Trained);

% (e) Calculate the MSE for each model on the test data (in mTestData.txt ).
mTest = load('data/mTestData.txt');
Y_Test = mTest(: ,1); X_Test = mTest(: ,2);

% Quadratic model
X_Test_2 = polyx(X_Test,2);
MSE_Quadratic_Test = mse(regress_learner,X_Test_2, Y_Test);
fprintf('The MSE for the quadratic predictor on test data was: %.2f\n', MSE_Quadratic_Test);

% Septic model
X_Test_7 = polyx(X_Test,7);
MSE_Septic_Test = mse(septic_learner,X_Test_7, Y_Test);
fprintf('The MSE for the septic predictor on test data was: %.2f\n', MSE_Septic_Test);

% (f) Calculate the MAE for each model on the test data. Compare the
% obtained MAE values with the MSE values obtained in above (e).

% Quadratic model
X_Test_2 = polyx(X_Test,2);
MAE_Quadratic_Test = mae(regress_learner,X_Test_2, Y_Test);
fprintf('The MAE for the quadratic predictor on test data was: %.2f\n', MAE_Quadratic_Test);

% Septic model
X_Test_7 = polyx(X_Test,7);
MAE_Septic_Test = mae(septic_learner,X_Test_7, Y_Test);
fprintf('The MAE for the septic predictor on test data was: %.2f\n', MAE_Septic_Test);

% Compare them how!!!!!!!


%% 2. kNN Regression
% Define the values of K to test
K_Values = [1, 2, 3, 5, 10, 50];

% Plot training data
figure('name', 'kNN Regression predictor');
plot (Xtr, Ytr, 'bo');
legend('Training Data','location','Northwest');
xlim([0 1]);
ylim([0 4.5]);
xlabel('X');
ylabel('Y');
hold on

xline = [0:.01:2]' ; % transpose : make a column vector , like training x

% Create a kNN regression predictor from the data Xtr, Ytr for each K.
for i=K_Values
    learner = knnRegress(i, Xtr, Ytr);
    
    % Plot the current models predictions
    yline = predict(learner, xline);
    plot(xline, yline, '', 'DisplayName', strcat('K=', num2str(i)));
end

%% 3. Hold-out and Cross-validation

% (a) compute the MSE of the test data on a model trained on only the first
% 20 training data examples for k = 1, 2, 3, . . . , 100. Plot both train
% and test MSE versus k on a log-log scale first 20 training points

X_Training_20 = Xtr(1:20, :); Y_Training_20 = Ytr(1:20, :);

% Matrix for storing the results
MSE_Training_Matrix = zeros(100,3);
MSE_Test_Matrix = zeros(100, 3);

for k=1:100
    
    % Create model based on the first 20 training samples
    learner_20 = knnRegress(k, X_Training_20, Y_Training_20);
    % Generate some predictions based on the previous model
    %Y_Prediction = predict(learner_20, X_Test);
    
    % Store the MSE of the training data
    MSE_Training_Matrix(k, 1) = mse(learner_20,Xtr, Ytr);  % not sure if this is what is required
    % Store the MSE of the predictions
    MSE_Test_Matrix(k, 1) = mse(learner_20,X_Test, Y_Test);
    
    % Create a learner for all of the data
    learner_all_training = knnRegress(k, Xtr, Ytr);    
    
    % Store the MSE of the training data
    MSE_Training_Matrix(k, 2) = mse(learner_all_training,Xtr, Ytr);  % not sure if this is what is required
    % Store the MSE of the predictions
    MSE_Test_Matrix(k, 2) = mse(learner_all_training,X_Test, Y_Test);
    
    
   
    Cross_Val_Training_MSE = 0;
    Cross_Val_Testing_MSE = 0;
    for i=1:4
        
        % Find the index to split the training data set
        start_index = 20*(i - 1) + 1;       % 1:20,21:40,41:60,61:80
        end_index = start_index + 19;
        Test_Data_Range = start_index:end_index;
        
        % Split the training data
        Cross_Val_Testing_X = Xtr(Test_Data_Range);
        Cross_Val_Testing_Y = Ytr(Test_Data_Range);
        Cross_Val_Training_X = Xtr(setdiff(1:80, Test_Data_Range));
        Cross_Val_Training_Y = Ytr(setdiff(1:80, Test_Data_Range));
        
        % Create a learner using this data
        learner_Cross_Val = knnRegress(k, Cross_Val_Training_X, Cross_Val_Training_Y);
        
        % Calculate the mse
        training_mse = mse(learner_Cross_Val,Cross_Val_Training_X, Cross_Val_Training_Y);
        testing_mse = mse(learner_Cross_Val,Cross_Val_Testing_X, Cross_Val_Testing_Y);
        
        Cross_Val_Training_MSE = Cross_Val_Training_MSE + training_mse;
        Cross_Val_Testing_MSE = Cross_Val_Testing_MSE + training_mse;

    end
    
    MSE_Training_Matrix(k, 3) = Cross_Val_Training_MSE / 4.0;
    MSE_Test_Matrix(k, 3) = Cross_Val_Testing_MSE / 4.0;
end

% Plot training data
figure('name', 'kNN Training MSE');

loglog(1:100, MSE_Training_Matrix(:, 1),'r');
hold on;
loglog(1:100, MSE_Training_Matrix(:, 2),'b');
loglog(1:100, MSE_Training_Matrix(:, 3),'g');
grid on
xlabel('K');
ylabel('Mean Squared Error');
legend('20 Training Data Points', 'All Training Data Points', 'Cross Validation');

% Plot Test data
figure('name', 'kNN Test MSE');

loglog(1:100, MSE_Test_Matrix(:, 1),'r');
hold on;
grid on;
loglog(1:100, MSE_Test_Matrix(:, 2),'b');
loglog(1:100, MSE_Test_Matrix(:, 3),'g');

xlabel('K');
ylabel('Mean Squared Error');
legend('20 Training Data Points', 'All Training Data Points', 'Cross Validation');


