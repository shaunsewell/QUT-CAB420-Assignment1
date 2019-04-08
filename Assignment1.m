%% CAB420 Assignment 1 
%   Shaun Sewell N9509623 
%
%% 1. Features, Classes, and Linear Regression

% (a) Plot the training data in a scatter plot.
mTrain = load('data/mTrainData.txt'); % Load training data
Xtr = mTrain(: ,1); Ytr = mTrain(: ,2); % Separate features

figure('name', 'Scatter Plot of Training Data');
plot (Xtr, Ytr, 'bo'); % Plot training data
xlabel('X');
ylabel('Y');
title('Scatter Plot of Training Data');

%Create a linear regression learner using the above functions. Plot it on
%the same plot as the training data.

regression_learner = linearReg(Xtr ,Ytr); % train a linear regression learner
predictor = predict(regression_learner , Xtr); % use it for prediction
xline = 0:.01:2 ; % transpose : make a column vector , like training x
yline = predict( regression_learner , polyx (xline ,2) ); % assuming quadratic features

figure('name', 'Linear Regression predictor');
plot(xline, yline, 'ro');
hold on % Plot training data and label figure.
plot (xtr, ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Linear Regression predictor');