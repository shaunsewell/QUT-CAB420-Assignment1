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
legend('Training Data');
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
plot(xline, yline, 'r*');
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
plot(xline, yline, 'r*');
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
plot (Xtr, Ytr, 'ks');
title('kNN Regression predictor')
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
    plot(xline, yline, '', 'DisplayName', strcat('K=', num2str(i)),'LineWidth',2);
end

%% 3. Hold-out and Cross-validation

% (a) compute the MSE of the test data on a model trained on only the first
% 20 training data examples for k = 1, 2, 3, . . . , 100. Plot both train
% and test MSE versus k on a log-log scale first 20 training points
X_Training_20 = Xtr(1:20, :); Y_Training_20 = Ytr(1:20, :);
MSE_Training_20 = zeros(100,1);
MSE_Test_20 = zeros(100,1);

for k=1:100
    learner_20 = knnRegress(k, X_Training_20, Y_Training_20);
    MSE_Training_20(k, 1) = mse(learner_20,Xtr, Ytr);
    % Store the MSE of the predictions
    MSE_Test_20(k, 1) = mse(learner_20,X_Test, Y_Test);
end

figure('name', 'MSE Comparison 20');
loglog(1:100, MSE_Training_20(:, 1),'r', 'LineWidth', 1.5);
hold on;
loglog(1:100, MSE_Test_20(:, 1),'--r', 'LineWidth', 1.5);
title('MSE using limited training set');
xlabel('K');
ylabel('Mean Squared Error');
legend('Training', 'Test');
hold off;

% (b) Repeat, but use all the training data. What happened? Contrast with
% your results from Problem 1 (hint: which direction is ?complexity? in
% this picture?).

MSE_Training_All = zeros(100,1);
MSE_Test_All = zeros(100,1);

for k=1:100
    learner_All = knnRegress(k, Xtr, Ytr);
    MSE_Training_All(k, 1) = mse(learner_All,Xtr, Ytr);
    % Store the MSE of the predictions
    MSE_Test_All(k, 1) = mse(learner_All,X_Test, Y_Test);
end

figure('name', 'MSE Comparison');
loglog(1:100, MSE_Training_All(:, 1),'b', 'LineWidth', 1.5);
hold on;
loglog(1:100, MSE_Test_All(:, 1),'--b', 'LineWidth', 1.5);
title('MSE using entire training set');
xlabel('K');
ylabel('Mean Squared Error');
legend('Training', 'Test');
hold off;


%(c) Using ?only the training data,? estimate the curve using 4-fold
%cross-validation. Split the training data into two parts, indices 1:20 and
%21:140; use the larger of the two as training data and the smaller as
%testing data, then repeat three more times with different sets of 20 and
%average the MSE. Plot this together with (a) and (b). Use different colors
%or marks to differentiate three scenarios. Discus why might we need to use
%this technique via comparing curves of three scenario?



% Matrix for storing the results
MSE_Training_CV = zeros(100,1);
MSE_Test_CV = zeros(100, 1);

for k=1:100
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
    
    MSE_Training_CV(k, 1) = Cross_Val_Training_MSE / 4.0;
    MSE_Test_CV(k, 1) = Cross_Val_Testing_MSE / 4.0;
end



% Plot training data
figure('name', 'kNN MSE Comparison');
loglog(1:100, MSE_Training_20(:, 1),'r', 'LineWidth', 1.5);
hold on;
loglog(1:100, MSE_Test_20(:, 1),'--r', 'LineWidth', 1.5);
loglog(1:100, MSE_Training_All(:, 1),'b', 'LineWidth', 1.5);
loglog(1:100, MSE_Test_All(:, 1),'--b', 'LineWidth', 1.5);
loglog(1:100, MSE_Training_CV(:, 1),'g', 'LineWidth', 1.5);
loglog(1:100, MSE_Test_CV(:, 1),'--k', 'LineWidth', 2);
grid on
title('kNN MSE Comparison');
xlabel('K');
ylabel('Mean Squared Error');
legend('20 Training Points', '20 Training Points Test', 'All Training Points', 'All Training Points Test', 'Cross Validation', 'Cross Validation Test', 'Location', 'southeast');
hold off;
%% 4. Nearest Neighbor Classifiers 
% (a) Plot the data by their feature values, using the class value to
% select the color.

% Load the data
iris = load('data/iris.txt');
pi = randperm(size(iris, 1));
Y = iris(pi, 5); X = iris(pi, 1:2);

% Plot the feature values
figure('Name','Feature Values of Iris Dataset');
hold on;
colours = unique(Y);
for colour = 1:length(colours)
    
    RGB_Vector = [0, 0, 0];
    RGB_Vector(colour) = 1;
    feature_indicies = find(Y==colours(colour));
    feature_points = X(feature_indicies, :);
    feature_point_x = feature_points(1:end, 1);
    feature_point_y = feature_points(1:end, 2);
    scatter(feature_point_x, feature_point_y, [], RGB_Vector, 'filled');
end

legend('Class 0', 'Class 1', 'Class 2');
title('Feature Values of Iris Dataset');
hold off;

% (b) Use the provided knnClassify class to learn a 1-nearest-neighbor
% predictor. Use the function class2DPlot(learner,X,Y)? to plot the
% decision regions and training data together.

iris_knn_learner_1 = knnClassify(1, X, Y);
class2DPlot(iris_knn_learner_1,X,Y);
title('1-nearest-neighbour predictor');

% (c) Do the same thing for several values of k (say, [3, 10, 30]) and
% comment on their appearance.

iris_k_values_small = [3, 10, 30];

for k = iris_k_values_small
    iris_knn_learner = knnClassify(k, X, Y);
    class2DPlot(iris_knn_learner,X,Y);
    title(strcat(int2str(k), '-nearest-neighbour predictor'));
end
%%
%


% d) Now split the data into an 80/20 training/validation split. For k =
% [1, 2, 5, 10, 50, 100, 200], learn a model on the 80% and calculate its
% performance (# of data classified incorrectly) on the validation data.

% Split X and Y into training and Validation sets
iris_training_x = X(1:118, 1:end);  % 118 is approx 80%
iris_training_y = Y(1:118);

iris_val_x = X(119:end, 1:end);
iris_val_y = Y(119:end);

% Define the K values to test
iris_k_values_large = [1, 2, 5, 10, 50, 100, 200];
errors = [];

for k = iris_k_values_large
    % Train a model
    iris_knn_learner = knnClassify(k, iris_training_x, iris_training_y); 
    % Make some predictions
    prediction = predict(iris_knn_learner, iris_val_x); 
    % Measure the errors in the predictions
    errors = [errors, numel(find(prediction~=iris_val_y))];   
    %hold on;
end

figure('Name','Performace of k'); 
plot(iris_k_values_large, errors,'-*')
title('Performance of k');
xlabel('Value of k')
ylabel('Number of incorrect predictions')

%% 5. Perceptrons and Logistic Regression

% Load the data
iris = load('data/iris.txt');
X = iris(:,1:2); Y=iris(:,end); 
[X, Y] = shuffleData(X,Y); % Randomise
X = rescale(X);
%Class 0 vs 1
XA = X(Y<2,:); YA=Y(Y<2);
%Class 1 vs 2
XB = X(Y>0,:); YB=Y(Y>0);

% (a) Show the two classes in a scatter plot and verify that one is
% linearly separable while the other is not.

figure('Name','Class 0 vs 1');
hold on;
% find members of Class 0 and plot them
index_class_zero = find(YA==0);
x_points_class_zero = XA(index_class_zero, 1:end);
scatter(x_points_class_zero(:, 1), x_points_class_zero(:, 2), 'filled');

% Find memebers of Class 1 and plot them
index_class_one = find(YA==1);
x_points_class_one = XA(index_class_one, 1:end);
scatter(x_points_class_one(:, 1), x_points_class_one(:, 2), 'r', 'filled');
title('Class 0 vs 1');
legend('Class 0', 'Class 1', 'FontSize', 12);
hold off;

figure('Name','Class 1 vs 2');
hold on;

% Already have class one so just need to plot it
scatter(x_points_class_one(:, 1), x_points_class_one(:, 2), 'r', 'filled');

% Find memebers of Class 2 and plot them
index_class_two = find(YB==2);
x_points_class_two = XB(index_class_two, 1:end);
scatter(x_points_class_two(:, 1), x_points_class_two(:, 2), 'filled');
title('Class 1 vs 2');
legend('Class 1', 'Class 2', 'FontSize', 12);
hold off;


% (b) Write ( fill in) the functio?n ?@logisticClassify2/plot2DLinear.m so
% that it plots the two classes of data in different colors, along with the
% decision boundary (a line). Include the listing of your code in your
% report. To demo your function plot the decision boundary corresponding to
% the classifier sign(.5 + 1x 1 + .25x 2) along with the A data, and again
% with the B data.

learnerB=logisticClassify2(); % create "blank" learner
learnerB=setClasses(learnerB, unique(YA)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; 
learnerB=setWeights(learnerB, wts); % set the learner's parameters

plot2DLinear(learnerB, XA, YA);
title('A Data with Decision Boundary');

plot2DLinear(learnerB, XB, YB);
title('B Data with Decision Boundary');

% (c) Complete the ?predict.m function to make predictions for your linear
% classifier. Again, verify that your function works by computing &
% reporting the error rate of the classifier in the previous part on both
% data sets A and B. (The error rate on data set A should be 0.0505.)

% Data set A
Y_Predictions_A = predict(learnerB,XA);

classification_error_A = numel(find(Y_Predictions_A~=YA));
final_classifiaction_error_A = classification_error_A/size(YA,1); % = 0.0505
disp(strcat({'The error rate for class A is:'},...
    {' '},{num2str(final_classifiaction_error_A,' %.4f')}));

Y_Predictions_B = predict(learnerB,XB);

classification_error_B = numel(find(Y_Predictions_B~=YB));
final_classifiaction_error_B = classification_error_B/size(YA,1);
disp(strcat({'The error rate for class A is:'},...
    {' '},{num2str(final_classifiaction_error_B,' %.4f')}));

% (e) Complete your ?train.m ?function to perform stochastic gradient
% descent on the logistic loss function.

% (f) Run your logistic regression classifier on both data sets (A and B);
% for this problem, use no regularization (? = 0). Describe your parameter
% choices (stepsize, etc.) and show a plot of both the convergence of the
% surrogate loss and error rate, and a plot of the final converged
% classifier with the data

learner_A=logisticClassify2(); % create "blank" learner
learner_A=setClasses(learner_A, unique(YA)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; 
learner_A=setWeights(learner_A, wts); % set the learner's parameters

learner_A = train(learner_A, XA, YA);

figure();
plotClassify2D(learner_A, XA, YA);
legend('Setosa', 'Versicolour', 'FontSize', 12);
xlabel('Sepal Length')
ylabel('Sepal Width')

new_YB = YB - 1;
learner_B=logisticClassify2(); % create "blank" learner
learner_B=setClasses(learner_B, unique(YA)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; 
learner_B=setWeights(learner_B, wts); % set the learner's parameters
learner_B = train(learner_B, XB, new_YB);

% Plot final converged classifier decision boundaries.
figure();
plotClassify2D(learner_B, XB, new_YB);
legend('Class 1', 'Class 2', 'FontSize', 12);

% (g)

